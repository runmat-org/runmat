const MUTATING_MAP_METHODS = new Set(["clear", "delete", "set"]);

export function deepImmutable(value, seen = new WeakMap()) {
  if (value === null || typeof value !== "object") return value;
  if (seen.has(value)) return seen.get(value);
  if (value instanceof Map) {
    const target = new Map();
    const view = new Proxy(target, {
      get(map, property) {
        if (MUTATING_MAP_METHODS.has(property)) {
          return () => { throw new TypeError("validated authority maps are immutable"); };
        }
        const member = Reflect.get(map, property, map);
        return typeof member === "function" ? member.bind(map) : member;
      },
    });
    seen.set(value, view);
    for (const [key, entry] of value) target.set(deepImmutable(key, seen), deepImmutable(entry, seen));
    return Object.freeze(view);
  }
  if (Array.isArray(value)) {
    const copy = [];
    seen.set(value, copy);
    for (const entry of value) copy.push(deepImmutable(entry, seen));
    return Object.freeze(copy);
  }
  const copy = {};
  seen.set(value, copy);
  for (const [key, entry] of Object.entries(value)) copy[key] = deepImmutable(entry, seen);
  return Object.freeze(copy);
}
