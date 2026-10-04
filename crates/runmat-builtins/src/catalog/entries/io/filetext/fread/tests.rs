use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn number() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

#[test]
fn default_read_is_double_matrix_and_count_is_double_scalar() {
    let inferred = inference::infer(
        &request(vec![number()], vec![LiteralValue::Unknown], 2),
        &FREAD_CATALOG_ENTRY,
    );
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(inferred.outputs.len(), 2);
    assert_eq!(inferred.outputs[0].shape, ShapeFact::Ranked { rank: 2 });
    assert_eq!(inferred.outputs[0].kind, number().kind);
    assert_eq!(inferred.outputs[1], number());
    assert!(inferred.effects.0.contains(&EffectKind::FilesystemRead));
    assert!(inferred
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
}

#[test]
fn literal_precision_and_unknown_like_do_not_guess_dimensions_or_class() {
    let char_precision = ValueFact::scalar(ValueKindFact::Character);
    let known = inference::infer(
        &request(
            vec![number(), char_precision.clone()],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Character("*char".into()),
            ],
            1,
        ),
        &FREAD_CATALOG_ENTRY,
    );
    assert_eq!(known.outputs[0].kind, ValueKindFact::Character);
    assert_eq!(known.outputs[0].shape, ShapeFact::Ranked { rank: 2 });
    let integer = inference::infer(
        &request(
            vec![number(), number(), char_precision.clone()],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String("*uint64".into()),
            ],
            1,
        ),
        &FREAD_CATALOG_ENTRY,
    );
    assert!(matches!(
        integer.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            ..
        })
    ));
    let dynamic = inference::infer(
        &request(
            vec![
                number(),
                char_precision,
                ValueFact::unknown(DynamicReason::RuntimeValue),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("*uint64".into()),
                LiteralValue::Unknown,
            ],
            1,
        ),
        &FREAD_CATALOG_ENTRY,
    );
    assert!(matches!(dynamic.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn precision_aliases_repeats_and_character_forms_share_runtime_grammar() {
    for (label, expected) in [
        (
            "REAL*4",
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
        ),
        (
            "3*uint16=>ushort",
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt16,
                domain: NumericDomain::Real,
            }),
        ),
        (
            "*integer*8",
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Int64,
                domain: NumericDomain::Real,
            }),
        ),
        ("2*uint8=>char", ValueKindFact::Character),
    ] {
        let inferred = inference::infer(
            &request(
                vec![number(), ValueFact::scalar(ValueKindFact::String)],
                vec![LiteralValue::Unknown, LiteralValue::String(label.into())],
                1,
            ),
            &FREAD_CATALOG_ENTRY,
        );
        assert_eq!(inferred.outputs[0].kind, expected, "{label}");
    }
    let invalid = inference::infer(
        &request(
            vec![number(), ValueFact::scalar(ValueKindFact::String)],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("3*uint16=>notaclass".into()),
            ],
            1,
        ),
        &FREAD_CATALOG_ENTRY,
    );
    assert!(matches!(invalid.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn invalid_fid_and_arity_are_diagnosed_without_false_unknown_error() {
    let bad = inference::infer(
        &request(
            vec![ValueFact::scalar(ValueKindFact::String)],
            vec![LiteralValue::Unknown],
            1,
        ),
        &FREAD_CATALOG_ENTRY,
    );
    assert!(!bad.diagnostics.is_empty());
    let unknown = inference::infer(
        &request(
            vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
            vec![LiteralValue::Unknown],
            1,
        ),
        &FREAD_CATALOG_ENTRY,
    );
    assert!(unknown.diagnostics.is_empty());
    let empty = inference::infer(&request(Vec::new(), Vec::new(), 1), &FREAD_CATALOG_ENTRY);
    assert!(!empty.diagnostics.is_empty());
}
