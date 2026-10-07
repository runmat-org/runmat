use super::*;
use crate::{builtin_catalog_entry_by_name, BuiltinInferenceRule};
use runmat_types::{
    CallRequest, CapabilityRequirement, LiteralContext, OutputSelection, RequestedOutputCount,
    ValueFact, ValueKindFact,
};

#[test]
fn entries_are_canonical_and_use_typed_family_rules() {
    let mkdir = builtin_catalog_entry_by_name("mkdir").expect("mkdir catalog entry");
    let rmdir = builtin_catalog_entry_by_name("rmdir").expect("rmdir catalog entry");
    assert_eq!(mkdir.identity.name, MKDIR_CATALOG_ENTRY.identity.name);
    assert_eq!(rmdir.identity.name, RMDIR_CATALOG_ENTRY.identity.name);
    assert!(matches!(
        mkdir.contract.inference_rule,
        BuiltinInferenceRule::Io(crate::IoInferenceRule::ReplFs(
            crate::IoReplFsInferenceRule::Directory(crate::DirectoryInferenceRule::Lifecycle(
                crate::DirectoryLifecycleInferenceRule::Create
            ))
        ))
    ));
    assert!(matches!(
        rmdir.contract.inference_rule,
        BuiltinInferenceRule::Io(crate::IoInferenceRule::ReplFs(
            crate::IoReplFsInferenceRule::Directory(crate::DirectoryInferenceRule::Lifecycle(
                crate::DirectoryLifecycleInferenceRule::Remove
            ))
        ))
    ));
}

#[test]
fn inference_returns_logical_status_and_character_diagnostics() {
    for entry in [&MKDIR_CATALOG_ENTRY, &RMDIR_CATALOG_ENTRY] {
        assert_eq!(
            entry.contract.capabilities,
            &[CapabilityRequirement::Filesystem]
        );
        let request = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::String)],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(3)),
        };
        let inference = crate::infer_catalog_call(entry, &request);
        assert!(inference
            .capabilities
            .0
            .contains(&CapabilityRequirement::Filesystem));
        assert_eq!(inference.outputs.len(), 3);
        assert!(matches!(inference.outputs[0].kind, ValueKindFact::Logical));
        assert!(matches!(
            inference.outputs[1].kind,
            ValueKindFact::Character
        ));
        assert!(matches!(
            inference.outputs[2].kind,
            ValueKindFact::Character
        ));
        assert!(inference.diagnostics.is_empty());
    }
}

#[test]
fn text_shapes_reject_only_proven_non_rows() {
    for entry in [&MKDIR_CATALOG_ENTRY, &RMDIR_CATALOG_ENTRY] {
        for (dimensions, invalid) in [
            (vec![Some(1), Some(3), Some(1)], false),
            (vec![Some(1), Some(1), Some(3)], true),
            (vec![None, Some(3), None], false),
            (vec![Some(2), Some(3)], true),
        ] {
            let request = CallRequest {
                arguments: vec![ValueFact::proven(
                    ValueKindFact::Character,
                    runmat_types::ShapeFact::from(dimensions),
                    runmat_types::StorageFact::Dense,
                )],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
            };
            assert_eq!(
                !crate::infer_catalog_call(entry, &request)
                    .diagnostics
                    .is_empty(),
                invalid
            );
        }
    }
}
