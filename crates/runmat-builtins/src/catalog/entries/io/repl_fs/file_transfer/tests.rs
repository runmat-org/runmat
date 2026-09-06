use super::*;
use crate::{builtin_catalog_entry_by_name, BuiltinInferenceRule};
use runmat_types::{
    CallRequest, LiteralContext, OutputSelection, RequestedOutputCount, ValueFact, ValueKindFact,
};

#[test]
fn entries_are_canonical_and_use_typed_family_rules() {
    for (name, expected) in [
        ("copyfile", FileTransferInferenceRule::Copy),
        ("movefile", FileTransferInferenceRule::Move),
    ] {
        let entry = builtin_catalog_entry_by_name(name).expect("catalog entry");
        assert!(matches!(
            entry.contract.inference_rule,
            BuiltinInferenceRule::Io(crate::IoInferenceRule::ReplFs(
                crate::IoReplFsInferenceRule::File(crate::FileInferenceRule::Transfer(rule))
            )) if rule == expected
        ));
    }
}

#[test]
fn inference_returns_double_status_and_character_diagnostics() {
    for entry in [&COPYFILE_CATALOG_ENTRY, &MOVEFILE_CATALOG_ENTRY] {
        let request = CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(3)),
        };
        let inference = crate::infer_catalog_call(entry, &request);
        assert_eq!(inference.outputs.len(), 3);
        assert!(matches!(
            inference.outputs[0].kind,
            ValueKindFact::Numeric(runmat_types::NumericFact {
                class: runmat_types::NumericClass::Double,
                domain: runmat_types::NumericDomain::Real
            })
        ));
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
