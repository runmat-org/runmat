/** Provider errors retain their browser error identity across the JS/WASM
 * boundary. HTTP-style status remains available to workspace consumers. */
export function invalidInput(message: string): Error & { status: number } {
  return withStatus(new Error(message), 400);
}

export function notFound(path: string, entry: "Path" | "File" | "Directory" = "Path"): Error & { status: number } {
  return withStatus(new DOMException(`${entry} not found: ${path}`, "NotFoundError"), 404);
}

export function alreadyExists(path: string): Error & { status: number } {
  return withStatus(new Error(`Path already exists: ${path}`), 409);
}

function withStatus(error: Error, status: number): Error & { status: number } {
  return Object.assign(error, { status });
}
