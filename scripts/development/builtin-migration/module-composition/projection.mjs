import { compareCodePoint } from "../constants.mjs";
import { array, enumValue, exact, kind, stableId } from "../schema.mjs";
import {
  parseCompositionChild, parseModuleCompositionProjection,
} from "./schema.mjs";

export function applyModuleCompositionTransitions(baselineValue, transitionValues) {
  const baseline = parseModuleCompositionProjection(structuredClone(baselineValue));
  const products = new Map(baseline.products.map((entry) => [entry.product_id, structuredClone(entry)]));
  const transitionIds = new Set();
  for (const value of transitionValues) {
    const transition = parseTransition(value, products);
    if (transitionIds.has(transition.transition_id)) throw new Error(`duplicate module composition transition ${transition.transition_id}`);
    transitionIds.add(transition.transition_id);
    applyTransition(products, transition);
  }
  return parseModuleCompositionProjection({
    schema_version: 2,
    kind: "runmat-builtin-module-composition-projection",
    products: [...products.values()].sort((left, right) => compareCodePoint(left.product_id, right.product_id)),
  });
}

export function parseModuleCompositionTransition(value, projectionValue) {
  const projection = parseModuleCompositionProjection(projectionValue);
  return parseTransition(value, new Map(projection.products.map((entry) => [entry.product_id, entry])));
}

function parseTransition(value, products) {
  kind(value, 2, "runmat-builtin-module-composition-transition", "module composition transition");
  exact(value, ["schema_version", "kind", "transition_id", "changes"], "module composition transition");
  const transitionId = stableId(value.transition_id, "module composition transition id");
  const changes = array(value.changes, `${transitionId} composition changes`).map((entry) => parseChange(entry, products, transitionId));
  const keys = changes.map(changeKey);
  if (new Set(keys).size !== keys.length) throw new Error(`${transitionId}: composition changes must be unique`);
  if (JSON.stringify(keys) !== JSON.stringify([...keys].sort(compareCodePoint))) throw new Error(`${transitionId}: composition changes must use canonical code-point order`);
  return { ...value, transition_id: transitionId, changes };
}

function parseChange(value, products, transitionId) {
  exact(value, ["product_id", "operation", "before", "after"], `${transitionId} composition change`);
  const productId = stableId(value.product_id, `${transitionId} product id`);
  const product = products.get(productId);
  if (!product) throw new Error(`${transitionId}: unknown composition product ${productId}`);
  const operation = enumValue(value.operation, ["add", "remove", "replace"], `${transitionId} operation`);
  const context = {
    productId, crateRole: product.crate_role, productPath: product.path,
    modulePath: product.module_path, aggregations: product.aggregations,
  };
  const before = value.before === null ? null : parseCompositionChild(value.before, context);
  const after = value.after === null ? null : parseCompositionChild(value.after, context);
  if ((operation === "add") !== (before === null) || (operation === "remove") !== (after === null)) {
    throw new Error(`${transitionId}: ${operation} change has inconsistent before/after state`);
  }
  if (operation === "replace" && (before === null || after === null || before.module !== after.module)) {
    throw new Error(`${transitionId}: replacement must preserve the exact child module key`);
  }
  if (operation === "replace" && JSON.stringify(before) === JSON.stringify(after)) {
    throw new Error(`${transitionId}: composition replacement must change the reviewed child`);
  }
  return { ...value, product_id: productId, operation, before, after };
}

function applyTransition(products, transition) {
  for (const change of transition.changes) {
    const product = products.get(change.product_id);
    const children = new Map(product.children.map((entry) => [entry.module, entry]));
    const module = (change.after ?? change.before).module;
    const current = children.get(module) ?? null;
    if (JSON.stringify(current) !== JSON.stringify(change.before)) throw new Error(`${transition.transition_id}: ${change.product_id}/${module} prior state differs from the effective projection`);
    if (change.after === null) children.delete(module);
    else children.set(module, structuredClone(change.after));
    product.children = [...children.values()].sort((left, right) => compareCodePoint(left.module, right.module));
  }
}

function changeKey(entry) { return `${entry.product_id}\0${(entry.after ?? entry.before).module}`; }
