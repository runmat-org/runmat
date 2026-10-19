//! Static inference for findedge.

use super::super::shared::{arity, double_fact, graph_operand, node_selector};
use crate::{catalog::inference::finish_fixed, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest, ShapeFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, &mut diagnostics, 3, 3);
    graph_operand(request, &mut diagnostics, false);
    node_selector(request, &mut diagnostics, 1);
    node_selector(request, &mut diagnostics, 2);
    finish_fixed(entry, request, double_fact(ShapeFact::Scalar), diagnostics)
}
