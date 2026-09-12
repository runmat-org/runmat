import { nonempty } from "../schema.mjs";

export function gitTreeOid(value, label) {
  if (typeof value !== "string" || !/^[a-f0-9]{40,64}$/.test(value)) {
    throw new Error(`${label} is invalid`);
  }
  return value;
}

export function signerFingerprint(value, label) {
  const result = nonempty(value, label);
  if (!/^[A-Za-z0-9:+/=_-]{16,160}$/.test(result)) throw new Error(`${label} is invalid`);
  return result;
}
