import { createHash } from "node:crypto";
import { compareCodePoint } from "./constants.mjs";

export function evidenceDigest(value) {
  return `sha256:${createHash("sha256").update(canonicalJson(value)).digest("hex")}`;
}

export function canonicalJson(value) {
  return encode(value, new Set(), "$");
}

function encode(value, ancestors, path) {
  if (value === null || typeof value === "string" || typeof value === "boolean") return JSON.stringify(value);
  if (typeof value === "number") {
    if (!Number.isFinite(value) || Object.is(value, -0)) throw new Error(`Unsupported numeric evidence at ${path}`);
    return JSON.stringify(value);
  }
  if (typeof value !== "object") throw new Error(`Unsupported evidence value at ${path}`);
  if (ancestors.has(value)) throw new Error(`Cyclic evidence value at ${path}`);
  ancestors.add(value);
  let encoded;
  if (Array.isArray(value)) {
    const own = Reflect.ownKeys(value).filter((key) => key !== "length");
    if (own.length !== value.length || own.some((key, index) => key !== String(index))) throw new Error(`Sparse or extended evidence array at ${path}`);
    encoded = `[${value.map((entry, index) => encode(entry, ancestors, `${path}[${index}]`)).join(",")}]`;
  } else {
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) throw new Error(`Unsupported evidence object at ${path}`);
    const own = Reflect.ownKeys(value);
    if (own.some((key) => typeof key !== "string" || !Object.prototype.propertyIsEnumerable.call(value, key))) {
      throw new Error(`Unsupported evidence property at ${path}`);
    }
    const fields = own.sort(compareCodePoint);
    encoded = `{${fields.map((key) => `${JSON.stringify(key)}:${encode(value[key], ancestors, `${path}.${key}`)}`).join(",")}}`;
  }
  ancestors.delete(value);
  return encoded;
}
