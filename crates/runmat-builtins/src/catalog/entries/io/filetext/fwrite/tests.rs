use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}

fn numeric_scalar(class: NumericClass, domain: NumericDomain) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact { class, domain }))
}

#[test]
fn binary_write_infers_double_count_and_filesystem_effects() {
    let inferred = inference::infer(
        &request(vec![
            numeric_scalar(NumericClass::Double, NumericDomain::Real),
            numeric_scalar(NumericClass::UInt64, NumericDomain::Real),
        ]),
        &FWRITE_CATALOG_ENTRY,
    );
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );
    assert!(inferred.effects.0.contains(&EffectKind::FilesystemWrite));
    assert!(inferred
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
}

#[test]
fn definite_bad_id_data_precision_and_arity_are_diagnosed() {
    let invalid = inference::infer(
        &request(vec![
            ValueFact::scalar(ValueKindFact::String),
            numeric_scalar(NumericClass::Double, NumericDomain::Complex),
            ValueFact::scalar(ValueKindFact::Logical),
        ]),
        &FWRITE_CATALOG_ENTRY,
    );
    assert!(invalid.diagnostics.len() >= 3);
    let arity = inference::infer(&request(Vec::new()), &FWRITE_CATALOG_ENTRY);
    assert!(!arity.diagnostics.is_empty());
}

#[test]
fn unknown_controls_do_not_receive_guessed_static_diagnostics() {
    let inferred = inference::infer(
        &request(vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            ValueFact::unknown(DynamicReason::RuntimeValue),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ]),
        &FWRITE_CATALOG_ENTRY,
    );
    assert!(inferred.diagnostics.is_empty());
}

#[test]
fn character_precision_requires_proven_single_row_without_guessing_unknown_rows() {
    let fid = numeric_scalar(NumericClass::Double, NumericDomain::Real);
    let data = numeric_scalar(NumericClass::Double, NumericDomain::Real);
    let invalid = inference::infer(
        &request(vec![
            fid.clone(),
            data.clone(),
            ValueFact {
                kind: ValueKindFact::Character,
                shape: ShapeFact::Shaped {
                    dims: vec![DimensionFact::Known(2), DimensionFact::Known(3)],
                },
                ..ValueFact::unknown(DynamicReason::RuntimeValue)
            },
        ]),
        &FWRITE_CATALOG_ENTRY,
    );
    assert!(!invalid.diagnostics.is_empty());
    let unknown = inference::infer(
        &request(vec![
            fid,
            data,
            ValueFact {
                kind: ValueKindFact::Character,
                shape: ShapeFact::Ranked { rank: 2 },
                ..ValueFact::unknown(DynamicReason::RuntimeValue)
            },
        ]),
        &FWRITE_CATALOG_ENTRY,
    );
    assert!(unknown.diagnostics.is_empty());
}

#[test]
fn character_precision_rejects_known_pages_but_preserves_unknown_and_singleton_pages() {
    let fid = numeric_scalar(NumericClass::Double, NumericDomain::Real);
    let data = numeric_scalar(NumericClass::Double, NumericDomain::Real);
    let character = |dims| ValueFact {
        kind: ValueKindFact::Character,
        shape: ShapeFact::Shaped { dims },
        ..ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    let infer = |shape| {
        inference::infer(
            &request(vec![fid.clone(), data.clone(), character(shape)]),
            &FWRITE_CATALOG_ENTRY,
        )
    };
    assert!(!infer(vec![
        DimensionFact::Known(1),
        DimensionFact::Known(5),
        DimensionFact::Known(2),
    ])
    .diagnostics
    .is_empty());
    assert!(infer(vec![
        DimensionFact::Known(1),
        DimensionFact::Known(5),
        DimensionFact::Known(1),
    ])
    .diagnostics
    .is_empty());
    assert!(infer(vec![
        DimensionFact::Known(1),
        DimensionFact::Known(5),
        DimensionFact::Unknown,
    ])
    .diagnostics
    .is_empty());
}
