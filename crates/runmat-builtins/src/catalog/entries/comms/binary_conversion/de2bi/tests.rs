use super::*;
use runmat_types::*;

fn numeric(class: NumericClass, rows: usize, columns: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(columns)]),
        StorageFact::Dense,
    )
}

fn call(input: ValueFact, width: LiteralValue) -> CallRequest {
    CallRequest {
        arguments: vec![input, ValueFact::unknown(DynamicReason::RuntimeValue)],
        literals: LiteralContext::new(vec![LiteralValue::Unknown, width]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}

#[test]
fn input_class_and_known_row_count_survive_unknown_width() {
    let unknown_width = super::super::inference::infer_de2bi(
        &call(numeric(NumericClass::UInt16, 2, 3), LiteralValue::Unknown),
        &DE2BI_CATALOG_ENTRY,
    );
    assert!(unknown_width.diagnostics.is_empty());
    assert!(matches!(
        unknown_width.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt16,
            ..
        })
    ));
    assert_eq!(
        unknown_width.outputs[0].shape.known_dims(),
        Some(vec![Some(6), None])
    );
    assert_eq!(unknown_width.outputs[0].residency, ResidencyFact::Host);
    let known_width = super::super::inference::infer_de2bi(
        &call(
            numeric(NumericClass::Single, 1, 2),
            LiteralValue::Number(4.0),
        ),
        &DE2BI_CATALOG_ENTRY,
    );
    assert!(matches!(
        known_width.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            ..
        })
    ));
    assert_eq!(
        known_width.outputs[0].shape.known_dims(),
        Some(vec![Some(2), Some(4)])
    );
}

#[test]
fn invalid_controls_and_data_are_diagnosed_without_guessing_values() {
    let bad_width = super::super::inference::infer_de2bi(
        &call(
            numeric(NumericClass::Double, 1, 1),
            LiteralValue::Number(-2.0),
        ),
        &DE2BI_CATALOG_ENTRY,
    );
    assert!(bad_width
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DE2BI-CONTROL"));
    let invalid = super::super::inference::infer_de2bi(
        &call(
            ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::Scalar,
                StorageFact::Dense,
            ),
            LiteralValue::Unknown,
        ),
        &DE2BI_CATALOG_ENTRY,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DE2BI-DATA"));
    assert_eq!(DE2BI_CATALOG_ENTRY.bindings, &REQUIRED_DEFAULT_BINDING);
    assert_eq!(DE2BI_CATALOG_ENTRY.documentation.examples.len(), 5);
}
