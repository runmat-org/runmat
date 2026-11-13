use super::*;
use runmat_types::*;

fn request(args: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments: args,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn filename() -> ValueFact {
    ValueFact::scalar(ValueKindFact::String)
}

fn range() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    )
}

#[test]
fn default_and_native_result_facts_reflect_file_dependent_class() {
    let plain = inference::infer(
        &request(vec![filename()], vec![LiteralValue::Unknown], 2),
        &AUDIOREAD_CATALOG_ENTRY,
    );
    assert!(plain.diagnostics.is_empty(), "{:?}", plain.diagnostics);
    assert_eq!(plain.outputs[0].shape, ShapeFact::from(vec![None, None]));
    assert!(matches!(
        plain.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            ..
        })
    ));
    assert!(matches!(
        plain.outputs[1].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            ..
        })
    ));
    let native = inference::infer(
        &request(
            vec![filename(), ValueFact::scalar(ValueKindFact::String)],
            vec![LiteralValue::Unknown, LiteralValue::String("native".into())],
            1,
        ),
        &AUDIOREAD_CATALOG_ENTRY,
    );
    assert!(native.diagnostics.is_empty(), "{:?}", native.diagnostics);
    assert!(matches!(native.outputs[0].kind, ValueKindFact::Unknown));
    assert_eq!(native.outputs[0].residency, ResidencyFact::Host);
    assert_eq!(native.outputs[0].storage, StorageFact::Dense);
    assert_eq!(native.outputs[0].layout, LayoutFact::ColumnMajor);
}

#[test]
fn range_and_mode_controls_reject_only_proven_invalid_inputs() {
    let integer_range = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int32,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    assert!(!inference::infer(
        &request(vec![filename(), integer_range], vec![], 1),
        &AUDIOREAD_CATALOG_ENTRY
    )
    .diagnostics
    .is_empty());
    let good = inference::infer(
        &request(
            vec![
                filename(),
                range(),
                ValueFact::scalar(ValueKindFact::String),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String("double".into()),
            ],
            2,
        ),
        &AUDIOREAD_CATALOG_ENTRY,
    );
    assert!(good.diagnostics.is_empty(), "{:?}", good.diagnostics);
    let unknown = inference::infer(
        &request(
            vec![filename(), ValueFact::unknown(DynamicReason::RuntimeValue)],
            vec![],
            1,
        ),
        &AUDIOREAD_CATALOG_ENTRY,
    );
    assert!(unknown.diagnostics.is_empty(), "{:?}", unknown.diagnostics);
    let bad_mode = inference::infer(
        &request(
            vec![filename(), ValueFact::scalar(ValueKindFact::String)],
            vec![LiteralValue::Unknown, LiteralValue::String("int16".into())],
            1,
        ),
        &AUDIOREAD_CATALOG_ENTRY,
    );
    assert!(!bad_mode.diagnostics.is_empty());
}
