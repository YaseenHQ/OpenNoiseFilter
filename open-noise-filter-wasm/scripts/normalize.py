# Copyright 2026 YaseenHQ
# SPDX-License-Identifier: Apache-2.0

"""Rebuild a lowered model with outputs reordered (cache outs first, spec last).

onnx2c's codegen is sensitive to graph-output ordering on this graph: with the
original order [spec_out, cache_out_*] it emits code that diverges from ORT;
the rebuilt order [cache_out_*, spec_out] compiles exact (validated 1.1e-5).
This makes the working order a deterministic build step.

Usage: python scripts/normalize.py <in.onnx> <out.onnx>
"""
import sys
import onnx
from onnx import helper

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
outs = [o for o in g.output if o.name.startswith("cache_out")] + \
       [o for o in g.output if not o.name.startswith("cache_out")]
g2 = helper.make_graph(list(g.node), "norm", list(g.input), outs, list(g.initializer))
m2 = helper.make_model(g2, opset_imports=list(m.opset_import))
m2.ir_version = m.ir_version
onnx.checker.check_model(m2)
onnx.save(m2, dst)
print(f"normalized {src} -> {dst}, outputs: {[o.name for o in outs]}")
