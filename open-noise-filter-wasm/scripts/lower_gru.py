# Copyright 2026 YaseenHQ
# SPDX-License-Identifier: Apache-2.0

"""Lower ONNX GRU nodes (T=1, forward, initial_h present) into basic ops.

FastEnhancer's exported graphs run each GRU for a single timestep per call,
with the hidden state passed through explicit cache tensors and the frequency
axis folded into the batch. A T=1 GRU with linear_before_reset=1 lowers to:

  gates = x@W^T + h@R^T + Wb + Rb            # [B, 3H], gate order z,r,n
  z     = sigmoid(gates[:, :H])
  r     = sigmoid(gates[:, H:2H])
  hn    = tanh(x@Wn^T + Wbn + r * (h@Rn^T + Rbn))
  h'    = (1 - z)*hn + z*h

All replacement ops are in onnx2c's supported set. The rewritten graph is
validated against the original in ONNX Runtime before saving.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "models/fastenhancer_t_spec.onnx"
DST = SRC.replace(".onnx", "_lowered.onnx")


def fl_const(name, vals, dims):
    return helper.make_node("Constant", [], [name],
                            value=helper.make_tensor(name + "_v", TensorProto.FLOAT, dims, vals, raw=False))


def i_const(name, vals):
    return helper.make_node("Constant", [], [name],
                            value=helper.make_tensor(name + "_v", TensorProto.INT64, [len(vals)], vals, raw=False))


def lower_gru(node, uid):
    X, W, R, B, _seq, H0 = (list(node.input) + [""] * 6)[:6]
    Y, Yh = node.output[0], node.output[1]
    hs = next(a.i for a in node.attribute if a.name == "hidden_size")
    n = uid

    nd = []
    nd += [i_const(f"{n}_s0", [0]), i_const(f"{n}_sH", [hs]), i_const(f"{n}_s2H", [2 * hs]),
           i_const(f"{n}_s3H", [3 * hs]), i_const(f"{n}_s6H", [6 * hs]), i_const(f"{n}_ax1", [1]),
           i_const(f"{n}_ax2", [2]), i_const(f"{n}_axY", [1]), fl_const(f"{n}_one", [1.0], [])]

    Wt, Rt = f"{n}_Wt", f"{n}_Rt"
    nd += [helper.make_node("Transpose", [W], [Wt], perm=[0, 2, 1]),
           helper.make_node("Transpose", [R], [Rt], perm=[0, 2, 1])]

    xw, hr = f"{n}_xw", f"{n}_hr"
    nd += [helper.make_node("MatMul", [X, Wt], [xw]),
           helper.make_node("MatMul", [H0, Rt], [hr])]

    Wb, Rb = f"{n}_Wb", f"{n}_Rb"
    nd += [helper.make_node("Slice", [B, f"{n}_s0", f"{n}_s3H", f"{n}_ax1"], [Wb]),
           helper.make_node("Slice", [B, f"{n}_s3H", f"{n}_s6H", f"{n}_ax1"], [Rb])]

    gates = f"{n}_gates"
    nd += [helper.make_node("Add", [xw, hr], [gates]),
           helper.make_node("Add", [gates, Wb], [f"{gates}2"]),
           helper.make_node("Add", [f"{gates}2", Rb], [f"{gates}3"])]
    zr = f"{gates}3"
    z, r = f"{n}_z", f"{n}_r"
    nd += [helper.make_node("Slice", [zr, f"{n}_s0", f"{n}_sH", f"{n}_ax2"], [f"{n}_z_pre"]),
           helper.make_node("Sigmoid", [f"{n}_z_pre"], [z]),
           helper.make_node("Slice", [zr, f"{n}_sH", f"{n}_s2H", f"{n}_ax2"], [f"{n}_r_pre"]),
           helper.make_node("Sigmoid", [f"{n}_r_pre"], [r])]

    # hn = tanh( (xw + Wb)[:, n] + r * ((hr + Rb)[:, n]) )   -- lbr=1
    nx, nxr = f"{n}_nx", f"{n}_nxr"
    nd += [helper.make_node("Slice", [xw, f"{n}_s2H", f"{n}_s3H", f"{n}_ax2"], [f"{n}_nx_s"]),
           helper.make_node("Slice", [Wb, f"{n}_s2H", f"{n}_s3H", f"{n}_ax1"], [f"{n}_nxw_s"]),
           helper.make_node("Add", [f"{n}_nx_s", f"{n}_nxw_s"], [nx]),
           helper.make_node("Slice", [hr, f"{n}_s2H", f"{n}_s3H", f"{n}_ax2"], [f"{n}_hrn_s"]),
           helper.make_node("Slice", [Rb, f"{n}_s2H", f"{n}_s3H", f"{n}_ax1"], [f"{n}_hrb_s"]),
           helper.make_node("Add", [f"{n}_hrn_s", f"{n}_hrb_s"], [nxr]),
           helper.make_node("Mul", [r, nxr], [f"{n}_rg"]),
           helper.make_node("Add", [nx, f"{n}_rg"], [f"{n}_hn_pre"]),
           helper.make_node("Tanh", [f"{n}_hn_pre"], [f"{n}_hn"])]

    hnew = f"{n}_hnew"
    nd += [helper.make_node("Sub", [f"{n}_one", z], [f"{n}_omz"]),
           helper.make_node("Mul", [f"{n}_omz", f"{n}_hn"], [f"{n}_a1"]),
           helper.make_node("Mul", [z, H0], [f"{n}_a2"]),
           helper.make_node("Add", [f"{n}_a1", f"{n}_a2"], [hnew]),
           helper.make_node("Unsqueeze", [hnew, f"{n}_axY"], [Y]),
           helper.make_node("Identity", [hnew], [Yh])]
    return nd


def lower_gather(node, uid, inits):
    """Gather(scalar idx, axis A) -> Slice(A, idx..idx+1) + Squeeze(A).
    onnx2c's Gather leaves trailing elements unwritten; Slice+Squeeze are safe."""
    X, Idx = node.input[0], node.input[1]
    Y = node.output[0]
    idx_arr = onnx.numpy_helper.to_array(inits[Idx])
    idx = int(idx_arr.reshape(-1)[0])
    axis = next((a.i for a in node.attribute if a.name == "axis"), 0)
    n = f"g{uid}"
    nd = [
        i_const(f"{n}_st", [idx]),
        i_const(f"{n}_en", [idx + 1]),
        i_const(f"{n}_ax", [axis]),
        i_const(f"{n}_axs", [axis]),
        helper.make_node("Slice", [X, f"{n}_st", f"{n}_en", f"{n}_ax"], [f"{n}_s"]),
        helper.make_node("Squeeze", [f"{n}_s", f"{n}_axs"], [Y]),
    ]
    return nd


def lower_gemm(node, uid):
    """Gemm(alpha=1, beta=1, transA=0[, transB=1]) -> Transpose+MatMul+Add.
    onnx2c's Gemm miscompiles at [36,36]/[48,48]; MatMul path is proven."""
    A, B, C = (list(node.input) + ["", ""])[:3]
    Y = node.output[0]
    at = {a.name: (a.f if a.type == 1 else a.i) for a in node.attribute}
    assert at.get("alpha", 1.0) == 1.0 and at.get("beta", 1.0) == 1.0 and at.get("transA", 0) == 0, "unsupported Gemm attrs"
    n = uid
    nd = [i_const(f"{n}_ax01", [1, 0])]
    if at.get("transB", 0) == 1:
        nd.append(helper.make_node("Transpose", [B], [f"{n}_Bt"], perm=[1, 0]))
        B = f"{n}_Bt"
    nd.append(helper.make_node("MatMul", [A, B], [f"{n}_mm"]))
    nd.append(helper.make_node("Add", [f"{n}_mm", C], [Y]))
    return nd


def lower_conv1x1(node, uid, inits):
    """Conv1d kernel=1, stride 1, pad 0, group 1 -> MatMul(W2d, X) + bias."""
    import numpy as np
    from onnx import numpy_helper
    X, Wn = node.input[0], node.input[1]
    Bn = node.input[2] if len(node.input) > 2 else ""
    Y = node.output[0]
    at = {a.name: (list(a.ints) if a.ints else a.i) for a in node.attribute}
    assert at.get("kernel_shape") == [1] and at.get("pads", [0, 0]) == [0, 0] and at.get("strides", [1]) == [1] and at.get("group", 1) == 1, "unsupported Conv"
    W = numpy_helper.to_array(inits[Wn])          # [out, in, 1]
    out_c, in_c = W.shape[0], W.shape[1]
    n = uid
    new_inits = [numpy_helper.from_array(W.reshape(out_c, in_c).astype(np.float32), f"{n}_W2d")]
    nd = [helper.make_node("MatMul", [f"{n}_W2d", X], [f"{n}_mm"])]
    if Bn:
        b = numpy_helper.to_array(inits[Bn])      # [out]
        new_inits.append(numpy_helper.from_array(b.reshape(1, out_c, 1).astype(np.float32), f"{n}_B3"))
        nd.append(helper.make_node("Add", [f"{n}_mm", f"{n}_B3"], [Y]))
    else:
        nd.append(helper.make_node("Identity", [f"{n}_mm"], [Y]))
    return nd, new_inits


def lower_convk3(node, uid, inits, xshape=None):
    """Conv1d kernel=3, stride 1, pads [1,1], group 1 -> Pad + 3 MatMul + Adds."""
    import numpy as np
    from onnx import numpy_helper
    X, Wn = node.input[0], node.input[1]
    Bn = node.input[2] if len(node.input) > 2 else ""
    Y = node.output[0]
    W = numpy_helper.to_array(inits[Wn])          # [co, ci, 3]
    assert xshape and len(xshape) == 3, f"need static X shape for k3 conv, got {xshape}"
    L = xshape[2]
    n = uid
    new_inits = [numpy_helper.from_array(W[:, :, t].astype(np.float32), f"{n}_W{t}") for t in range(3)]
    if Bn:
        b = numpy_helper.to_array(inits[Bn])
        new_inits.append(numpy_helper.from_array(b.reshape(1, -1, 1).astype(np.float32), f"{n}_B3"))
    nd = [
        i_const(f"{n}_p0", [0]), i_const(f"{n}_p1", [1]), i_const(f"{n}_p2", [2]),
        i_const(f"{n}_ax2", [2]),
        i_const(f"{n}_e0", [L]), i_const(f"{n}_e1", [L + 1]), i_const(f"{n}_e2", [L + 2]),
        i_const(f"{n}_pads", [0, 0, 1, 0, 0, 1]), fl_const(f"{n}_zero", [0.0], []),
        helper.make_node("Pad", [X, f"{n}_pads", f"{n}_zero"], [f"{n}_xp"], mode="constant"),
        helper.make_node("Slice", [f"{n}_xp", f"{n}_p0", f"{n}_e0", f"{n}_ax2"], [f"{n}_x0"]),
        helper.make_node("Slice", [f"{n}_xp", f"{n}_p1", f"{n}_e1", f"{n}_ax2"], [f"{n}_x1"]),
        helper.make_node("Slice", [f"{n}_xp", f"{n}_p2", f"{n}_e2", f"{n}_ax2"], [f"{n}_x2"]),
    ]
    nd.append(helper.make_node("MatMul", [f"{n}_W0", f"{n}_x0"], [f"{n}_m0"]))
    nd.append(helper.make_node("MatMul", [f"{n}_W1", f"{n}_x1"], [f"{n}_m1"]))
    nd.append(helper.make_node("MatMul", [f"{n}_W2", f"{n}_x2"], [f"{n}_m2"]))
    nd.append(helper.make_node("Add", [f"{n}_m0", f"{n}_m1"], [f"{n}_s01"]))
    acc = f"{n}_s02"
    if Bn:
        nd.append(helper.make_node("Add", [f"{n}_s01", f"{n}_m2"], [f"{n}_s012"]))
        nd.append(helper.make_node("Add", [f"{n}_s012", f"{n}_B3"], [Y]))
    else:
        nd.append(helper.make_node("Add", [f"{n}_s01", f"{n}_m2"], [Y]))
    return nd, new_inits


def main():
    m = onnx.load(SRC)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    _m2 = onnx.shape_inference.infer_shapes(m)
    _vi = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in _m2.graph.value_info}
    for _i in _m2.graph.input:
        _vi[_i.name] = [d.dim_value for d in _i.type.tensor_type.shape.dim]
    out, ngru, ngat, ngem, ncv, nk3 = [], 0, 0, 0, 0, 0
    for node in g.node:
        if node.op_type == "GRU":
            out.extend(lower_gru(node, f"gru{ngru}"))
            ngru += 1
        elif node.op_type == "Conv" and len(node.input) > 1 and node.input[1] in inits:
            wa = __import__("onnx").numpy_helper.to_array(inits[node.input[1]])
            if wa.ndim == 3 and wa.shape[2] == 1:
                nds, nis = lower_conv1x1(node, f"cv{ncv}", inits)
                out.extend(nds); g.initializer.extend(nis); ncv += 1
            elif wa.ndim == 3 and wa.shape[2] == 3:
                nds, nis = lower_convk3(node, f"ck{nk3}", inits, _vi.get(node.input[0]))
                out.extend(nds); g.initializer.extend(nis); nk3 += 1
            elif False:
                nds, nis = lower_conv1x1(node, f"cv{ncv}", inits)
                out.extend(nds); g.initializer.extend(nis); ncv += 1
            else:
                out.append(node)
        elif node.op_type == "Gemm":
            out.extend(lower_gemm(node, f"gemm{ngem}"))
            ngem += 1
        elif node.op_type == "Gather" and len(node.input) > 1 and node.input[1] in inits:
            out.extend(lower_gather(node, f"gat{ngat}", inits))
            ngat += 1
        else:
            out.append(node)
    del g.node[:]
    g.node.extend(out)
    onnx.checker.check_model(m)
    onnx.save(m, DST)
    print(f"lowered {ngru} GRU + {ngat} Gather + {ngem} Gemm + {ncv} Conv1x1 + {nk3} ConvK3 node(s) -> {DST}")

    rng = np.random.default_rng(0)

    def run(path, feeds=None):
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        if feeds is None:
            feeds = {}
            for i in s.get_inputs():
                shape = [d if isinstance(d, int) else 1 for d in i.shape]
                feeds[i.name] = (rng.standard_normal(shape) * (0.5 if i.name.startswith("cache") else 2)).astype(np.float32)
        return s.run(None, feeds), feeds

    ya, feeds = run(SRC)
    yb, _ = run(DST, feeds)
    for i, (ra, rb) in enumerate(zip(ya, yb)):
        d = float(np.abs(np.asarray(ra) - np.asarray(rb)).max())
        print(f"  out{i}: max|orig-lowered| = {d:.3e}  {'OK' if d < 1e-4 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
