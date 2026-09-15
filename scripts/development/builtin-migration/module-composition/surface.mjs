import { childPathAttribute } from "./schema.mjs";

export function expectedCompositionSurface(product) {
  return {
    declarations: [...product.children].sort((left, right) => left.declaration_order - right.declaration_order).map((entry) => ({
      module: entry.module,
      visibility: entry.visibility,
      declaration_condition: entry.declaration_condition,
      path_attribute: childPathAttribute(product.path, entry),
      macro_use: entry.macro_use,
    })),
    reexports: product.children.flatMap((entry) => entry.reexports.map((reexport) => ({
      module: entry.module,
      visibility: reexport.visibility,
      condition: reexport.condition,
      doc_hidden: reexport.doc_hidden,
      reexport: reexport.kind === "glob"
        ? { kind: "glob" }
        : reexport.kind === "module"
          ? { kind: "module", alias: reexport.alias }
          : { kind: "named", items: reexport.items },
    }))),
    aggregation_exports: product.aggregation_exports,
    aggregations: product.crate_role === "runtime" ? [] : product.aggregations
      .filter((role) => !product.aggregation_exports.some((entry) => entry.role === role))
      .map((role) => ({
      role,
      children: product.children.flatMap((entry) => entry.aggregation_sources
        .filter((source) => source.role === role)
        .map((source) => ({
          source_kind: source.kind,
          module: entry.module,
          condition: source.condition,
          order: source.order,
        })))
        .sort((left, right) => left.order - right.order),
      })),
  };
}
