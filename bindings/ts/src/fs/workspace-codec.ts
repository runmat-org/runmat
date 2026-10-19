import { Encoder } from "cbor-x";
import { MAX_WORKSPACE_PUBLICATION_BYTES } from "../generated/fea-workspace-contracts.js";

// Standard maps and byte strings. JS typed-array extension tags cannot be
// consumed by Serde's internally tagged enum content deserializer.
const codec = new Encoder({ useRecords: false, tagUint8Array: false });
export const encodeWorkspace = (value: unknown): Uint8Array => codec.encode(value);
export const decodeWorkspace = (bytes: Uint8Array): unknown => codec.decode(bytes);

/** Enforce the wire budget while reading, before allocating the final payload. */
export async function readWorkspaceResponse(response: Response): Promise<Uint8Array> {
  const limit = 2 * MAX_WORKSPACE_PUBLICATION_BYTES + 8 * 1024 * 1024;
  const declared = Number(response.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > limit) {
    await response.body?.cancel();
    throw new Error("Workspace response exceeds byte budget");
  }
  if (!response.body) return new Uint8Array();
  const reader = response.body.getReader();
  let complete = false;
  try {
    let size = 0;
    const chunks: Uint8Array[] = [];
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      size += chunk.value.byteLength;
      if (size > limit) throw new Error("Workspace response exceeds byte budget");
      chunks.push(chunk.value);
    }
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    complete = true;
    return bytes;
  } finally {
    if (!complete) await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
