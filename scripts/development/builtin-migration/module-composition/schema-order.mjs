import { compareCodePoint } from "../constants.mjs";

export const AGGREGATION_ROLES = Object.freeze(["entries", "aliases", "constants"]);

export function validateAggregationOrders(children, roles, productId) {
  for (const role of roles) {
    const orders = children.flatMap((child) => child.aggregation_sources.filter((source) => source.role === role).map((source) => source.order));
    const expected = Array.from({ length: orders.length }, (_, index) => index);
    if (new Set(orders).size !== orders.length || JSON.stringify([...orders].sort((a, b) => a - b)) !== JSON.stringify(expected)) throw new Error(`${productId} ${role} aggregation orders must be unique and contiguous from zero`);
  }
}

export function validateDeclarationOrders(children, productId) {
  const orders = children.map((child) => child.declaration_order);
  const expected = Array.from({ length: orders.length }, (_, index) => index);
  if (new Set(orders).size !== orders.length || JSON.stringify([...orders].sort((a, b) => a - b)) !== JSON.stringify(expected)) {
    throw new Error(`${productId} declaration orders must be unique and contiguous from zero`);
  }
}

export function canonicalUnique(values, key, label, caseFold = false) {
  const keys = values.map(key);
  if (new Set(keys).size !== keys.length) throw new Error(`${label} must be unique`);
  if (caseFold && new Set(keys.map((entry) => entry.toLowerCase())).size !== keys.length) throw new Error(`${label} collide case-insensitively`);
  if (JSON.stringify(keys) !== JSON.stringify([...keys].sort(compareCodePoint))) throw new Error(`${label} must use canonical code-point order`);
}

export function caseFoldUnique(values, key, label) {
  const keys = values.map(key).map((entry) => entry.toLowerCase());
  if (new Set(keys).size !== keys.length) throw new Error(`${label} collide case-insensitively`);
}

export function aggregationRoleOrder(roles, label) {
  if (new Set(roles).size !== roles.length) throw new Error(`${label} must be unique`);
  const expected = AGGREGATION_ROLES.filter((role) => roles.includes(role));
  if (JSON.stringify(roles) !== JSON.stringify(expected)) throw new Error(`${label} must use entries, aliases, constants order`);
}
