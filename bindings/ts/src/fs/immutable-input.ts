/** Enforce the immutable manifest length before collecting the response bytes. */
export async function readImmutableFilesystemResponse(response: Response, size: number): Promise<Uint8Array> {
  if (!Number.isSafeInteger(size) || size < 0) throw new Error("Invalid immutable input size");
  if (!response.ok) throw new Error(`Immutable download failed (${response.status})`);
  const declared = response.headers.get("content-length");
  if (declared !== null && Number(declared) !== size) {
    await response.body?.cancel();
    throw new Error("Immutable input length differs from its manifest");
  }
  const reader = response.body?.getReader();
  if (!reader) {
    if (size === 0) return new Uint8Array();
    throw new Error("Immutable input length differs from its manifest");
  }
  let complete = false;
  try {
    const chunks: Uint8Array[] = [];
    let length = 0;
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      length += chunk.value.byteLength;
      if (length > size) throw new Error("Immutable input length differs from its manifest");
      chunks.push(chunk.value);
    }
    if (length !== size) throw new Error("Immutable input length differs from its manifest");
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
