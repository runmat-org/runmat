//! Static inference for numedges.

use super::super::shared::{arity, double_fact, graph_operand};
use crate::{catalog::inference::finish_fixed, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest, ShapeFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, &mut diagnostics, 1, 1);
    graph_operand(request, &mut diagnostics, false);
    finish_fixed(entry, request, double_fact(ShapeFact::Scalar), diagnostics)
}
