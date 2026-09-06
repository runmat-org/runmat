use super::*;
use crate::{BuiltinInferenceRule, IoInferenceRule, IoReplFsInferenceRule, PathInferenceRule};
use runmat_types::{
    CallRequest, LiteralContext, OutputSelection, RequestedOutputCount, ShapeFact, ValueFact,
    ValueKindFact,
};

#[test]
fn entries_are_identity_local_and_use_typed_rules() {
    for (entry, expected) in [
        (&ISFILE_CATALOG_ENTRY, PathPredicateInferenceRule::File),
        (&ISFOLDER_CATALOG_ENTRY, PathPredicateInferenceRule::Folder),
    ] {
        assert!(matches!(
            entry.contract.inference_rule,
            BuiltinInferenceRule::Io(IoInferenceRule::ReplFs(
                IoReplFsInferenceRule::Path(PathInferenceRule::Predicate(rule))
            )) if rule == expected
        ));
        assert_eq!(entry.documentation.examples.len(), 4);
    }
}

#[test]
fn inference_preserves_container_shape() {
    for entry in [&ISFILE_CATALOG_ENTRY, &ISFOLDER_CATALOG_ENTRY] {
        let request = CallRequest {
            arguments: vec![ValueFact::proven(
                ValueKindFact::String,
                ShapeFact::from(vec![Some(2), Some(3)]),
                runmat_types::StorageFact::Dense,
            )],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        };
        let inferred = crate::infer_catalog_call(entry, &request);
        assert!(inferred.diagnostics.is_empty());
        assert_eq!(inferred.outputs[0].shape, request.arguments[0].shape);
        assert!(matches!(inferred.outputs[0].kind, ValueKindFact::Logical));
    }
}
