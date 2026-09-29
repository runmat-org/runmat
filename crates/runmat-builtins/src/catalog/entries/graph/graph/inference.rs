//! Static inference for graph.

use super::super::shared::{
    adjacency_argument, arity, constructor_arguments, graph_object_fact, GRAPH_CLASS,
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
    finish_fixed(entry, request, graph_object_fact(GRAPH_CLASS), diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{
        DynamicReason, LiteralContext, OutputSelection, RequestedOutputCount, ShapeFact,
        StaticClassIdentity, StorageFact, ValueFact, ValueKindFact,
    };

    fn infer_arguments(arguments: Vec<ValueFact>) -> CallInference {
        crate::infer_catalog_call(
            &super::super::GRAPH_CATALOG_ENTRY,
            &CallRequest {
                arguments,
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
            },
        )
    }

    #[test]
    fn construction_proves_graph_class() {
        let result = infer_arguments(vec![]);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        let ValueKindFact::Object(ref object) = result.outputs[0].kind else {
            panic!("expected graph object");
        };
        assert_eq!(
            object.runtime_class.as_ref(),
            Some(&StaticClassIdentity::new("graph").owned())
        );
    }

    #[test]
    fn unresolved_constructor_role_does_not_become_a_false_adjacency_error() {
        let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
        let text = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::Scalar,
            StorageFact::Scalar,
        );
        let result = infer_arguments(vec![unknown.clone(), text.clone(), unknown]);
        assert!(!result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-GRAPH-ARITY"));
        let wrong = infer_arguments(vec![text.clone(), text]);
        assert!(wrong.diagnostics.is_empty(), "{:?}", wrong.diagnostics);
    }

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
        let result = infer_arguments(vec![
            adjacency,
            option,
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ]);
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-GRAPH-ARITY"));
    }
}
