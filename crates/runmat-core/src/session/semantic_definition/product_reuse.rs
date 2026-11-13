//! Same-stage reuse proof and session-publication reconciliation.

use std::collections::{HashMap, HashSet};

use runmat_hir::FunctionId;
use runmat_vm::{Bytecode, FunctionRegistry};

#[cfg(test)]
#[path = "product_reuse/tests.rs"]
mod tests;

/// Compare only this compilation's fully decorated definitions. The merged
/// executable also carries retained older bodies; those are not candidates.
pub(in crate::session) fn unchanged_product_remap(
    product: &Bytecode,
    current: &HashSet<FunctionId>,
    published: &FunctionRegistry,
    private_names: &HashSet<String>,
) -> Option<HashMap<FunctionId, FunctionId>> {
    if current.is_empty() {
        return None;
    }
    let functions = current
        .iter()
        .map(|id| Some((*id, product.function_registry.get_current(*id)?.clone())))
        .collect::<Option<HashMap<_, _>>>()?;
    let mut incoming = FunctionRegistry::new(functions);
    incoming.names = product
        .function_registry
        .names
        .iter()
        .filter(|(_, id)| current.contains(id))
        .map(|(name, id)| (name.clone(), *id))
        .collect();
    super::reuse::unchanged_closed_product_remap(product, &incoming, published, private_names)
}
