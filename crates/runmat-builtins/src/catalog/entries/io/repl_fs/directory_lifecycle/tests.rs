use super::*;
use crate::{builtin_catalog_entry_by_name, BuiltinInferenceRule};
use runmat_types::{
    CallRequest, LiteralContext, OutputSelection, RequestedOutputCount, ValueFact, ValueKindFact,
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
            crate::IoReplFsInferenceRule::DirectoryLifecycle(
                crate::DirectoryLifecycleInferenceRule::Create
            )
        ))
    ));
    assert!(matches!(
        rmdir.contract.inference_rule,
        BuiltinInferenceRule::Io(crate::IoInferenceRule::ReplFs(
            crate::IoReplFsInferenceRule::DirectoryLifecycle(
                crate::DirectoryLifecycleInferenceRule::Remove
            )
        ))
    ));
}

#[test]
fn inference_returns_logical_status_and_character_diagnostics() {
    for entry in [&MKDIR_CATALOG_ENTRY, &RMDIR_CATALOG_ENTRY] {
        let request = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::String)],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(3)),
        };
        let inference = crate::infer_catalog_call(entry, &request);
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
