/** Normalize the path form Git Bash can inject into a Windows environment. */
export function normalizeOnnx2cPath(value) {
  const gitMangle = /^(?:[A-Za-z]:[\\/])?Program Files[\\/]Git[\\/](mnt[\\/].*)$/.exec(value);
  return gitMangle ? `/${gitMangle[1].replace(/\\/g, "/")}` : value;
}

/** Convert a Windows path to its WSL mount path (D:\\x → /mnt/d/x). */
export function toWslPath(value) {
  const match = /^([A-Za-z]):[\\/](.*)$/.exec(value);
  if (!match) return value.replace(/\\/g, "/");
  return `/mnt/${match[1].toLowerCase()}/${match[2].replace(/\\/g, "/")}`;
}

/** Unix-style executable paths use WSL only when the build runs on Windows. */
export function onnx2cInvocation(executable, args, platform, { wslPaths = false } = {}) {
  const viaWsl = platform === "win32" && executable.startsWith("/") && !executable.startsWith("//");
  if (!viaWsl) return { command: executable, args, viaWsl: false };

  return {
    command: "wsl",
    args: ["--exec", executable, ...(wslPaths ? args.map(toWslPath) : args)],
    viaWsl: true,
  };
}
