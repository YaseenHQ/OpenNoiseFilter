import test from "node:test";
import assert from "node:assert/strict";
import { normalizeOnnx2cPath, onnx2cInvocation } from "./onnx2c-command.mjs";

test("native Unix absolute executable paths stay native", () => {
  const args = ["/tmp/build/model.onnx"];
  assert.deepEqual(onnx2cInvocation("/usr/local/bin/onnx2c", args, "linux", { wslPaths: true }), {
    command: "/usr/local/bin/onnx2c",
    args,
    viaWsl: false,
  });
});

test("Unix-style executable paths use WSL on Windows and translate file paths", () => {
  assert.deepEqual(onnx2cInvocation("/mnt/c/tools/onnx2c", ["D:\\build\\model.onnx"], "win32", { wslPaths: true }), {
    command: "wsl",
    args: ["--exec", "/mnt/c/tools/onnx2c", "/mnt/d/build/model.onnx"],
    viaWsl: true,
  });
});

test("native Windows executable paths preserve Windows arguments", () => {
  const args = ["D:\\build\\model.onnx"];
  assert.deepEqual(onnx2cInvocation("C:\\tools\\onnx2c.exe", args, "win32", { wslPaths: true }), {
    command: "C:\\tools\\onnx2c.exe",
    args,
    viaWsl: false,
  });
});

test("Git Bash path mangling is normalized for WSL", () => {
  assert.equal(normalizeOnnx2cPath("C:\\Program Files\\Git\\mnt\\c\\tools\\onnx2c"), "/mnt/c/tools/onnx2c");
});

test("Unix /mnt paths stay native on Unix", () => {
  const args = ["/mnt/d/build/model.onnx"];
  assert.deepEqual(onnx2cInvocation("/mnt/c/tools/onnx2c", args, "darwin", { wslPaths: true }), {
    command: "/mnt/c/tools/onnx2c",
    args,
    viaWsl: false,
  });
});

test("Windows drive paths containing mnt and UNC paths stay native", () => {
  const args = ["D:/build/model.onnx"];
  for (const executable of ["C:/mnt/tools/onnx2c.exe", "//server/share/onnx2c.exe", "\\\\server\\share\\onnx2c.exe"]) {
    assert.deepEqual(onnx2cInvocation(executable, args, "win32", { wslPaths: true }), {
      command: executable, args, viaWsl: false,
    });
  }
});
