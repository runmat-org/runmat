//! Static inference for digraph.

use super::super::shared::{
    adjacency_argument, arity, constructor_arguments, graph_object_fact, DIGRAPH_CLASS,
};
use crate::{catalog::inference::finish_fixed, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, &mut diagnostics, 0, 4);
    if request.arguments.len() == 1 {
        adjacency_argument(&request.arguments[0], &mut diagnostics);
    }
    if request.arguments.len() >= 2 {
        constructor_arguments(request, &mut diagnostics);
    }
    finish_fixed(
        entry,
        request,
        graph_object_fact(DIGRAPH_CLASS),
        diagnostics,
    )
}
