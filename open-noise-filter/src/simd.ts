/**
 * Shared WebAssembly SIMD probe — a minimal module using a v128 instruction;
 * validates only where WASM SIMD exists.
 */
export const SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1,
  8, 0, 65, 0, 253, 15, 253, 98, 11,
]);

export function isSimdSupported(): boolean {
  return typeof WebAssembly !== "undefined" && WebAssembly.validate(SIMD_PROBE);
}
