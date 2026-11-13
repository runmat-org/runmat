//! Static inference for shortestpath.

use super::super::shared::{arity, double_fact, graph_operand, node_list_fact, node_selector};
use crate::{catalog::inference::finish_fixed_outputs, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest, ShapeFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, &mut diagnostics, 3, 3);
    graph_operand(request, &mut diagnostics, false);
    node_selector(request, &mut diagnostics, 1);
    node_selector(request, &mut diagnostics, 2);
    finish_fixed_outputs(
        entry,
        request,
        vec![node_list_fact(), double_fact(ShapeFact::Scalar)],
        diagnostics,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{
        LiteralContext, NumericClass, NumericDomain, NumericFact, ObjectFact, OutputSelection,
        RequestedOutputCount, ShapeFact, StaticClassIdentity, StorageFact, ValueFact,
        ValueKindFact,
    };
    use std::collections::BTreeMap;

    #[test]
    fn two_result_form_rejects_three() {
        let graph = ValueFact::proven(
            ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(StaticClassIdentity::new("digraph").owned()),
                properties: BTreeMap::new(),
                properties_complete: false,
                handle_semantics: Some(false),
            }),
            ShapeFact::Scalar,
            StorageFact::Opaque,
        );
        let endpoint = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }));
        let args = vec![graph, endpoint.clone(), endpoint];
        let request = |outputs| CallRequest {
            arguments: args.clone(),
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        };
        let two = crate::infer_catalog_call(&super::super::SHORTESTPATH_CATALOG_ENTRY, &request(2));
        assert!(two.diagnostics.is_empty(), "{:?}", two.diagnostics);
        assert_eq!(two.outputs.len(), 2);
        let three =
            crate::infer_catalog_call(&super::super::SHORTESTPATH_CATALOG_ENTRY, &request(3));
        assert!(!three.diagnostics.is_empty());
    }
}
