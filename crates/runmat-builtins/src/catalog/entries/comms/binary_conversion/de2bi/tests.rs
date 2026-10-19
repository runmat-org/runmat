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
    let oversized = super::super::inference::infer_de2bi(
        &call(
            numeric(NumericClass::Double, 1, 1),
            LiteralValue::Number(10_000_001.0),
        ),
        &DE2BI_CATALOG_ENTRY,
    );
    assert!(oversized
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DE2BI-RESOURCE"));
    let false_width = super::super::inference::infer_de2bi(
        &call(
            numeric(NumericClass::Double, 1, 1),
            LiteralValue::Bool(false),
        ),
        &DE2BI_CATALOG_ENTRY,
    );
    assert!(false_width
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DE2BI-CONTROL"));
    let true_width = super::super::inference::infer_de2bi(
        &call(
            numeric(NumericClass::Double, 1, 1),
            LiteralValue::Bool(true),
        ),
        &DE2BI_CATALOG_ENTRY,
    );
    assert_eq!(
        true_width.outputs[0].shape.known_dims(),
        Some(vec![Some(1), Some(1)])
    );
    let zero_width = super::super::inference::infer_de2bi(
        &call(
            numeric(NumericClass::Double, 1, 1),
            LiteralValue::Number(0.0),
        ),
        &DE2BI_CATALOG_ENTRY,
    );
    assert!(zero_width
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DE2BI-CONTROL"));
    assert_eq!(
        zero_width.outputs[0].shape.known_dims(),
        Some(vec![Some(1), None])
    );
    let zero_integer_width = super::super::inference::infer_de2bi(
        &call(
            numeric(NumericClass::Double, 1, 1),
            LiteralValue::Integer {
                text: "0".to_string(),
                class: NumericClass::UInt8,
            },
        ),
        &DE2BI_CATALOG_ENTRY,
    );
    assert!(zero_integer_width
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DE2BI-CONTROL"));
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
    assert_eq!(DE2BI_CATALOG_ENTRY.documentation.examples.len(), 7);
}

#[test]
fn positional_controls_diagnose_proven_misplacement() {
    let input = numeric(NumericClass::Double, 1, 2);
    for (arguments, literals, expected, position) in [
        (
            3,
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("left-msb".into()),
                LiteralValue::Number(2.0),
            ],
            "RM-DE2BI-ORDER",
            1,
        ),
        (
            3,
            vec![
                LiteralValue::Unknown,
                LiteralValue::Number(3.0),
                LiteralValue::Empty,
            ],
            "RM-DE2BI-CONTROL",
            2,
        ),
        (
            4,
            vec![
                LiteralValue::Unknown,
                LiteralValue::Number(3.0),
                LiteralValue::Number(2.0),
                LiteralValue::Number(1.0),
            ],
            "RM-DE2BI-ORDER",
            3,
        ),
    ] {
        let request = CallRequest {
            arguments: std::iter::once(input.clone())
                .chain((1..arguments).map(|_| ValueFact::unknown(DynamicReason::RuntimeValue)))
                .collect(),
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        };
        let inferred = super::super::inference::infer_de2bi(&request, &DE2BI_CATALOG_ENTRY);
        assert!(
            inferred
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == expected
                    && diagnostic.argument == Some(position)),
            "{inferred:?}"
        );
    }
}
