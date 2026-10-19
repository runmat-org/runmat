use super::*;
use runmat_types::*;

fn signal(class: NumericClass, rows: usize, cols: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(cols)]),
        StorageFact::Dense,
    )
}

fn infer(input: ValueFact, literals: Vec<LiteralValue>) -> CallInference {
    let mut arguments = vec![
        input,
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
    ];
    arguments.extend((2..literals.len()).map(|_| ValueFact::unknown(DynamicReason::RuntimeValue)));
    crate::infer_catalog_call(
        &PSKMOD_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    )
}

#[test]
fn psk_symbol_shape_and_default_complex_double_are_known() {
    let result = infer(
        signal(NumericClass::Single, 2, 4),
        vec![LiteralValue::Unknown, LiteralValue::Number(8.0)],
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(4)])
    );
    assert_eq!(
        result.outputs[0].numeric(),
        Some(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        })
    );
}

#[test]
fn psk_bit_rows_and_single_output_are_option_dependent() {
    let result = infer(
        signal(NumericClass::UInt8, 4, 2),
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(4.0),
            LiteralValue::String("InputType".into()),
            LiteralValue::String("bit".into()),
            LiteralValue::String("OutputDataType".into()),
            LiteralValue::String("single".into()),
        ],
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(2)])
    );
    assert_eq!(
        result.outputs[0].numeric().map(|n| n.class),
        Some(NumericClass::Single)
    );
}

#[test]
fn psk_rejects_known_invalid_order_and_unsupported_wide_class() {
    let result = infer(
        signal(NumericClass::UInt64, 4, 1),
        vec![LiteralValue::Unknown, LiteralValue::Number(1.0)],
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
fn psk_rejects_logical_scalar_without_bit_input() {
    let result = infer(
        ValueFact::scalar(ValueKindFact::Logical),
        vec![LiteralValue::Unknown, LiteralValue::Number(4.0)],
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-MODULATION-LOGICAL"));
}

#[test]
fn psk_rejects_known_matrix_custom_symbol_order() {
    let result = crate::infer_catalog_call(
        &PSKMOD_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![
                signal(NumericClass::Double, 1, 4),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
                signal(NumericClass::Double, 2, 2),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Number(4.0),
                LiteralValue::Number(0.0),
                LiteralValue::Unknown,
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-MODULATION-MAPPING-SHAPE"));
}
