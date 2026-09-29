//! Static inference for predecessors.

use super::super::shared::{arity, graph_operand, node_list_fact, node_selector};
use crate::{catalog::inference::finish_fixed, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, &mut diagnostics, 2, 2);
    graph_operand(request, &mut diagnostics, false);
    node_selector(request, &mut diagnostics, 1);
    finish_fixed(entry, request, node_list_fact(), diagnostics)
}
