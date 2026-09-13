import fs from "node:fs";
import path from "node:path";

const ROOTS = new WeakMap();

export function openAuthorityRoot(target) {
  if (typeof target !== "string" || target.length === 0) {
    throw new TypeError("authority root must be a nonempty path");
  }
  const canonical = fs.realpathSync(path.resolve(target));
  const state = fs.lstatSync(canonical);
  if (state.isSymbolicLink() || !state.isDirectory()) {
    throw new Error("authority root must resolve to a real directory");
  }
  const root = Object.freeze({ path: canonical });
  ROOTS.set(root, directoryIdentity(state));
  return root;
}

export function assertAuthorityRoot(value) {
  if (!ROOTS.has(value)) throw new Error("operation requires an exact authority root");
  return value;
}

export function revalidateAuthorityRoot(value) {
  const root = assertAuthorityRoot(value);
  let state;
  try {
    state = fs.lstatSync(root.path);
  } catch (error) {
    throw new Error(`authority root cannot be revalidated: ${error.message}`);
  }
  const original = ROOTS.get(root);
  if (state.isSymbolicLink() || !state.isDirectory()
      || state.dev !== original.dev || state.ino !== original.ino) {
    throw new Error("authority root changed after observation");
  }
  return root;
}

function directoryIdentity(state) {
  return Object.freeze({ dev: state.dev, ino: state.ino });
}
