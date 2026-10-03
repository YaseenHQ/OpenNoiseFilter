"""Post-pass: rewrite onnx2c's MatMul bodies to SIMD-friendly ikj loop order.

onnx2c emits `for j { for k { Y[i][j] += A[i][k]*B[k][j] } }` — k innermost
strides through B's rows and defeats autovectorization. This rewrites each
generated node function to zero-fill Y then accumulate with j innermost
(contiguous in B's row and Y's row), which clang -O3 -msimd128 vectorizes.

Changes float summation order, so equivalence tolerance widens from ~1e-6
to ~1e-4 — re-run the specdiff after.

Usage: python scripts/fastmatmul.py <file.c>
"""
import re
import sys

path = sys.argv[1]
src = open(path).read()

func_re = re.compile(
    r"(FUNC_PREFIX void (node_\w*MatMul\w*)\(\s*const float (A\[[^)]*?\]),\s*const float (B\[[^)]*?\]),\s*float (Y\[[^)]*?\])\s*\)\s*\n\{\n)(.*?)(\n\}\n)",
    re.S,
)

dims_re = re.compile(r"\[(\d+)\]")


def rd(decl):
    """[1][48][128] -> (1,48,128)"""
    return tuple(int(x) for x in dims_re.findall(decl))


rewritten = 0
out = []
last = 0
for m in func_re.finditer(src):
    a, name, adecl, bdecl, ydecl, body, tail = m.groups()
    A, B, Y = rd(adecl), rd(bdecl), rd(ydecl)
    if len(A) not in (2, 3) or len(Y) != 3 or (len(A) == 3 and A[0] != 1) or Y[0] != 1 or len(B) not in (2, 3):
        out.append(src[last:m.end()])
        last = m.end()
        continue
    M, K, N = (A[-2], A[-1], Y[2])
    K_B = B[0] if len(B) == 2 else B[1]
    assert A[-1] == K_B and B[-1] == N and Y[1] == M, f"shape mismatch in {name}: A{A} B{B} Y{Y}"
    # flat restrict views over the original (untouched) parameters
    a_flat = "&A[0][0][0]" if len(A) == 3 else "&A[0][0]"
    b_flat = f"&B[0][0][0]" if len(B) == 3 else "&B[0][0]"
    y_flat = f"&Y[0][0][0]" if len(Y) == 3 else "&Y[0][0]"
    new_body = (
        f"\t/* MatMul (AbstractMatMul) — ikj reorder + flat restrict views for SIMD */\n"
        f"\t{{\n"
        f"\t\tconst float * __restrict__ ar = {a_flat};\n"
        f"\t\tconst float * __restrict__ br = {b_flat};\n"
        f"\t\tfloat * __restrict__ yr = {y_flat};\n"
        f"\t\tfor (unsigned i = 0; i < {M}; i++)\n"
        f"\t\tfor (unsigned j = 0; j < {N}; j++)\n"
        f"\t\t\tyr[i * {N} + j] = 0;\n"
        f"\t\tfor (unsigned k = 0; k < {K}; k++)\n"
        f"\t\tfor (unsigned i = 0; i < {M}; i++) {{\n"
        f"\t\t\tconst float a = ar[i * {K} + k];\n"
        f"\t\t\tconst float * __restrict__ brow = br + k * {N};\n"
        f"\t\t\tfloat * __restrict__ yrow = yr + i * {N};\n"
        f"\t\t\tfor (unsigned j = 0; j < {N}; j++)\n"
        f"\t\t\t\tyrow[j] += a * brow[j];\n"
        f"\t\t}}\n"
        f"\t}}\n"
    )
    out.append(src[last : m.start()])
    out.append(a + new_body + tail)
    last = m.end()
    rewritten += 1
out.append(src[last:])
open(path, "w").write("".join(out))
print(f"rewrote {rewritten} MatMul function(s) in {path}")
