use super::*;
use runmat_types::*;

fn input(class: NumericClass) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(4), Some(2)]),
        StorageFact::Dense,
    )
}

fn infer(value: ValueFact, literals: Vec<LiteralValue>) -> CallInference {
    let mut arguments = vec![
        value,
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
    ];
    arguments.extend((2..literals.len()).map(|_| ValueFact::unknown(DynamicReason::RuntimeValue)));
    crate::infer_catalog_call(
        &QAMMOD_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    )
}

#[test]
fn qam_single_input_preserves_class_and_shape() {
    let result = infer(
        input(NumericClass::Single),
        vec![LiteralValue::Unknown, LiteralValue::Number(16.0)],
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(4), Some(2)])
    );
    assert_eq!(
        result.outputs[0].numeric().map(|n| n.class),
        Some(NumericClass::Single)
    );
}

#[test]
fn qam_bit_rows_and_explicit_double_are_known() {
    let result = infer(
        input(NumericClass::UInt8),
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(16.0),
            LiteralValue::String("InputType".into()),
            LiteralValue::String("bit".into()),
            LiteralValue::String("OutputDataType".into()),
            LiteralValue::String("double".into()),
        ],
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );
    assert_eq!(
        result.outputs[0].numeric().map(|n| n.class),
        Some(NumericClass::Double)
    );
}

#[test]
fn qam_rejects_non_power_of_two_and_wide_integer_symbols() {
    let result = infer(
        input(NumericClass::Int32),
        vec![LiteralValue::Unknown, LiteralValue::Number(6.0)],
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-MODULATION-ORDER"));
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-MODULATION-CLASS"));
}

#[test]
fn qam_unknown_input_does_not_invent_output_precision() {
    let result = infer(
        ValueFact::unknown(DynamicReason::RuntimeValue),
        vec![LiteralValue::Unknown, LiteralValue::Number(16.0)],
    );
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}
