use crate::{
    BuiltinInferenceRule, IoInferenceRule, IoReplFsInferenceRule, TemporaryPathInferenceRule,
};
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn both_results_are_character_rows() {
    for name in ["tempdir", "tempname"] {
        let entry = crate::builtin_catalog_entry_by_name(name).expect("catalog entry");
        let result = crate::infer_catalog_call(entry, &request(Vec::new()));
        assert_eq!(result.outputs[0].kind, ValueKindFact::Character);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(1), None])
        );
        assert!(result.diagnostics.is_empty());
    }
}

#[test]
fn tempname_accepts_only_scalar_text_folders() {
    let entry = crate::builtin_catalog_entry_by_name("tempname").expect("catalog entry");
    let character_matrix = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    for invalid in [character_matrix, numeric] {
        assert_eq!(
            crate::infer_catalog_call(entry, &request(vec![invalid]))
                .diagnostics
                .len(),
            1
        );
    }
}

#[test]
fn family_rules_are_typed() {
    for (name, expected) in [
        ("tempdir", TemporaryPathInferenceRule::Directory),
        ("tempname", TemporaryPathInferenceRule::UniqueName),
    ] {
        let entry = crate::builtin_catalog_entry_by_name(name).expect("catalog entry");
        assert_eq!(
            entry.contract.inference_rule,
            BuiltinInferenceRule::Io(IoInferenceRule::ReplFs(
                IoReplFsInferenceRule::TemporaryPath(expected)
            ))
        );
    }
}
