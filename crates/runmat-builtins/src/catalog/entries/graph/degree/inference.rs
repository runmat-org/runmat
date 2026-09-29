//! Static inference for degree.

use super::super::shared::{arity, double_fact, graph_operand};
use crate::{catalog::inference::finish_fixed, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest, ShapeFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, &mut diagnostics, 1, 1);
    graph_operand(request, &mut diagnostics, false);
    finish_fixed(
        entry,
        request,
        double_fact(ShapeFact::from(vec![None, Some(1)])),
        diagnostics,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{
        LiteralContext, ObjectFact, OutputSelection, RequestedOutputCount, ShapeFact,
        StaticClassIdentity, StorageFact, ValueFact, ValueKindFact,
    };
    use std::collections::BTreeMap;

    #[test]
    fn known_other_class_is_rejected() {
        let wrong = ValueFact::proven(
            ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(StaticClassIdentity::new("tf").owned()),
                properties: BTreeMap::new(),
                properties_complete: false,
                handle_semantics: Some(false),
            }),
            ShapeFact::Scalar,
            StorageFact::Opaque,
        );
        let result = crate::infer_catalog_call(
            &super::super::DEGREE_CATALOG_ENTRY,
            &CallRequest {
                arguments: vec![wrong],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
            },
        );
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-GRAPH-OBJECT"));
    }
}
