import { inspectRepositoryRegularFile } from "./repository-state.mjs";

export function inspectEffectiveChildStates(repository, products) {
  return products.flatMap((product) => product.children.map((child) => ({
    product_id: product.product_id,
    module: child.module,
    source_kind: child.source_kind,
    source_path: child.source_path,
    ...inspectRepositoryRegularFile(
      repository, child.source_path,
      `${product.product_id}/${child.module}: effective child source`,
    ),
  })));
}

export function assertEffectiveChildStatesUnchanged(expected, observed) {
  if (JSON.stringify(observed) !== JSON.stringify(expected)) {
    throw new Error("effective child storage changed during materialization");
  }
}
