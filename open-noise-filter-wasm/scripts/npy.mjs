export function loadNpyF32(p) {
  const b = readFileSyncRef(p);
  if (b.subarray(0, 6).toString("latin1") !== "\x93NUMPY") throw new Error("not npy");
  const hl = b.readUInt16LE(8);
  const hdr = b.subarray(10, 10 + hl).toString("latin1");
  const m = hdr.match(/'shape':\s*\(([^)]*)\)/);
  const dims = m[1].split(",").map(s => s.trim()).filter(Boolean).map(Number);
  const n = dims.reduce((a, x) => a * x, 1);
  return new Float32Array(b.buffer, b.byteOffset + 10 + hl, n);
}
import { readFileSync as readFileSyncRef } from "fs";
