// Integer types on the FEA wire contracts, shared by the type renderer and the guards.
//
// JavaScript represents integers exactly only up to 2^53 - 1. Rust 64-bit and
// pointer-width integers must therefore carry the `runmat_types::wire_integer`
// serde adapter, which bounds them to that range on encode and decode; the
// renderer refuses any such field without it. Narrower integers keep their own
// range. Every integer renders as a named alias so guards check the exact range.
const MAX_WIRE_INTEGER = Number.MAX_SAFE_INTEGER;

const ALIASES = new Map([
  ["WireU8", [0, 255]], ["WireU16", [0, 65535]], ["WireU32", [0, 4294967295]],
  ["WireI8", [-128, 127]], ["WireI16", [-32768, 32767]], ["WireI32", [-2147483648, 2147483647]],
  ["WireU64", [0, MAX_WIRE_INTEGER]], ["WireI64", [-MAX_WIRE_INTEGER, MAX_WIRE_INTEGER]],
]);

const NARROW = new Map([
  ["u8", "WireU8"], ["u16", "WireU16"], ["u32", "WireU32"],
  ["i8", "WireI8"], ["i16", "WireI16"], ["i32", "WireI32"],
]);

const WIDE = new Map([["u64", "WireU64"], ["usize", "WireU64"], ["i64", "WireI64"], ["isize", "WireI64"]]);

const ADAPTER = "runmat_types::wire_integer";

function wideIntegerError(source) {
  return new Error(
    `64-bit integer ${source} on an FEA wire contract must carry #[serde(with = "${ADAPTER}[::option|::vec]")] ` +
      "so Rust and TypeScript share the 2^53 - 1 bound",
  );
}

// Render a field whose serde adapter is the wire-integer adapter, checking the
// adapter matches the field's Rust shape.
function renderAdaptedField(raw, adapter) {
  const shape = adapter.slice(ADAPTER.length);
  const pattern = { "": /^(\w+)$/, "::option": /^Option<\s*(\w+)\s*>$/, "::vec": /^Vec<\s*(\w+)\s*>$/ }[shape];
  const scalar = pattern && raw.replace(/\s+/g, "").match(pattern)?.[1];
  const alias = scalar && WIDE.get(scalar);
  if (!alias) throw new Error(`Wire-integer adapter ${adapter} does not match field type ${raw}`);
  return shape === "::vec" ? `${alias}[]` : alias;
}

function aliasDeclarations(body) {
  return [...ALIASES]
    .filter(([name]) => new RegExp(`\\b${name}\\b`).test(body))
    .map(([name, [min, max]]) => `/** Integer in [${min}, ${max}]; guards check Number.isSafeInteger and the range. */\nexport type ${name} = number;`)
    .join("\n");
}

function usedAliases(body) {
  return [...ALIASES.keys()].filter((name) => new RegExp(`\\b${name}\\b`).test(body));
}

// The guard call for an integer alias, or null for any other name.
function aliasGuard(name, value) {
  return ALIASES.has(name) ? `is${name}(${value})` : null;
}

// Guard helpers for the aliases a guard module uses.
function aliasGuardHelpers(names) {
  return [...ALIASES]
    .filter(([name]) => names.has(name))
    .map(([name, [min, max]]) => `function is${name}(value: unknown): boolean { return Number.isSafeInteger(value) && (value as number) >= ${min} && (value as number) <= ${max}; }`);
}

module.exports = { ADAPTER, MAX_WIRE_INTEGER, NARROW, WIDE, aliasDeclarations, aliasGuard, aliasGuardHelpers, renderAdaptedField, usedAliases, wideIntegerError };
