const SEP_PATTERN = /[\\/]+/;

export function normalizePath(input: string): string {
  if (!input) {
    return "/";
  }
  const parts = input.split(SEP_PATTERN);
  const stack: string[] = [];
  const absolute = input.startsWith("/");
  for (const raw of parts) {
    const part = raw.trim();
    if (!part || part === ".") {
      continue;
    }
    if (part === "..") {
      if (stack.length > 0) {
        stack.pop();
      }
      continue;
    }
    stack.push(part);
  }
  const joined = stack.join("/");
  if (absolute) {
    return "/" + joined;
  }
  return joined || ".";
}

export function dirname(path: string): string {
  const normalized = normalizePath(path);
  if (normalized === "/" || normalized === ".") {
    return normalized === "." ? "." : "/";
  }
  const parts = normalized.split("/");
  parts.pop();
  if (parts.length === 0) {
    return "/";
  }
  return "/" + parts.join("/");
}

export function basename(path: string): string {
  const normalized = normalizePath(path);
  if (normalized === "/" || normalized === ".") {
    return normalized;
  }
  const parts = normalized.split("/");
  return parts[parts.length - 1];
}

export function isSubPath(target: string, potentialParent: string): boolean {
  const normalizedTarget = normalizePath(target);
  const normalizedParent = normalizePath(potentialParent);
  if (normalizedParent === "/" || normalizedParent === ".") {
    return normalizedTarget !== normalizedParent;
  }
  return normalizedTarget.startsWith(normalizedParent.endsWith("/") ? normalizedParent : normalizedParent + "/");
}
