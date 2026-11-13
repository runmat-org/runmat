//! Static inference for distances.

use super::super::shared::{arity, double_fact, graph_operand, node_list_selector};
use crate::{catalog::inference::finish_fixed, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest, ShapeFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, &mut diagnostics, 1, 3);
    graph_operand(request, &mut diagnostics, false);
    node_list_selector(request, &mut diagnostics, 1);
    node_list_selector(request, &mut diagnostics, 2);
    finish_fixed(
        entry,
        request,
        double_fact(ShapeFact::from(vec![None, None])),
        diagnostics,
    )
}
