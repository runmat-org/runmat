const SEP_PATTERN = /[\\/]+/;
export function normalizePath(input) {
    if (!input) {
        return "/";
    }
    const parts = input.split(SEP_PATTERN);
    const stack = [];
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
export function dirname(path) {
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
export function basename(path) {
    const normalized = normalizePath(path);
    if (normalized === "/" || normalized === ".") {
        return normalized;
    }
    const parts = normalized.split("/");
    return parts[parts.length - 1];
}
export function isSubPath(target, potentialParent) {
    const normalizedTarget = normalizePath(target);
    const normalizedParent = normalizePath(potentialParent);
    if (normalizedParent === "/" || normalizedParent === ".") {
        return normalizedTarget !== normalizedParent;
    }
    return normalizedTarget.startsWith(normalizedParent.endsWith("/") ? normalizedParent : normalizedParent + "/");
}
//# sourceMappingURL=path.js.map