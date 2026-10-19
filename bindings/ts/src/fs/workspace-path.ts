export function canonicalIndexedDbPath(input: string): string {
  if (input.includes("\\") || input.includes("\0")) {
    throw invalidInput(`Invalid filesystem path: ${input}`);
  }
  const segments = input
    .split("/")
    .filter(Boolean);
  if (segments.some((segment) => segment === "." || segment === "..")) {
    throw invalidInput(`Invalid filesystem path: ${input}`);
  }
  return segments.length === 0 ? "/" : `/${segments.join("/")}`;
}

export function parentPath(path: string): string {
  const canonical = canonicalIndexedDbPath(path);
  const separator = canonical.lastIndexOf("/");
  return separator <= 0 ? "/" : canonical.slice(0, separator);
}

export function parentDirectories(path: string): string[] {
  const segments = canonicalIndexedDbPath(path).split("/").filter(Boolean);
  const parents: string[] = [];
  let current = "";
  for (const segment of segments.slice(0, -1)) {
    current += `/${segment}`;
    parents.push(current);
  }
  return parents;
}

export function isDescendant(path: string, ancestor: string): boolean {
  const prefix = ancestor === "/" ? "/" : `${ancestor}/`;
  return path !== ancestor && path.startsWith(prefix);
}

export function immediateChildName(path: string, directory: string): string | null {
  if (!isDescendant(path, directory)) {
    return null;
  }
  const offset = directory === "/" ? 1 : directory.length + 1;
  const remainder = path.slice(offset);
  return remainder && !remainder.includes("/") ? remainder : null;
}

export function invalidInput(message: string): Error & { status: number } {
  return withStatus(new Error(message), 400);
}

export function notFound(path: string): Error & { status: number } {
  return withStatus(new Error(`Path not found: ${path}`), 404);
}

export function alreadyExists(path: string): Error & { status: number } {
  return withStatus(new Error(`Path already exists: ${path}`), 409);
}

function withStatus(error: Error, status: number): Error & { status: number } {
  return Object.assign(error, { status });
}
