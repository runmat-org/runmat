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

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{
        LiteralContext, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact,
        ValueKindFact,
    };

    #[test]
    fn proven_logical_adjacency_with_two_options_is_invalid_arity() {
        let adjacency = ValueFact::proven(
            ValueKindFact::Logical,
            ShapeFact::from(vec![Some(2), Some(2)]),
            StorageFact::Dense,
        );
        let option = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::Scalar,
            StorageFact::Scalar,
        );
        let inference = crate::infer_catalog_call(
            &super::super::DIGRAPH_CATALOG_ENTRY,
            &CallRequest {
                arguments: vec![
                    adjacency,
                    option,
                    ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
                ],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
            },
        );
        assert!(inference
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-GRAPH-ARITY"));
    }
}
