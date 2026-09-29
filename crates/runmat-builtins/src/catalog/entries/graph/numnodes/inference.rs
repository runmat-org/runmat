//! Static inference for numnodes.

use super::super::shared::{arity, double_fact, graph_operand};
use crate::{catalog::inference::finish_fixed, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest, ShapeFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, &mut diagnostics, 1, 1);
    graph_operand(request, &mut diagnostics, false);
    finish_fixed(entry, request, double_fact(ShapeFact::Scalar), diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{
        LiteralContext, ObjectFact, OutputSelection, RequestedOutputCount, ResidencyFact,
        ShapeFact, StaticClassIdentity, StorageFact, ValueFact, ValueKindFact,
    };
    use std::collections::BTreeMap;

    #[test]
    fn count_is_host_scalar() {
        let graph = ValueFact::proven(
            ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(StaticClassIdentity::new("graph").owned()),
                properties: BTreeMap::new(),
                properties_complete: false,
                handle_semantics: Some(false),
            }),
            ShapeFact::Scalar,
            StorageFact::Opaque,
        );
        let result = crate::infer_catalog_call(
            &super::super::NUMNODES_CATALOG_ENTRY,
            &CallRequest {
                arguments: vec![graph],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
            },
        );
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
        assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    }
}
