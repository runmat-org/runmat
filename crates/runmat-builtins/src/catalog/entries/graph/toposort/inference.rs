//! Static inference for toposort.

use super::super::shared::{arity, graph_operand, node_list_fact};
use crate::{catalog::inference::finish_fixed, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, &mut diagnostics, 1, 1);
    graph_operand(request, &mut diagnostics, true);
    finish_fixed(entry, request, node_list_fact(), diagnostics)
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
    fn undirected_graph_is_rejected() {
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
            &super::super::TOPOSORT_CATALOG_ENTRY,
            &CallRequest {
                arguments: vec![graph],
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
