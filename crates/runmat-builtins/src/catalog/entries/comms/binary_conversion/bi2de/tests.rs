use super::*;
use runmat_types::*;

fn call(input: ValueFact, literals: Vec<LiteralValue>, extra: usize) -> CallRequest {
    let mut arguments = vec![input];
    arguments.extend((0..extra).map(|_| ValueFact::unknown(DynamicReason::RuntimeValue)));
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}

fn input(class: NumericClass, rows: usize, columns: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(columns)]),
        StorageFact::Dense,
    )
}

#[test]
fn class_and_shape_inference_does_not_guess_integer_widening() {
    let double = super::super::inference::infer_bi2de(
        &call(input(NumericClass::Double, 3, 4), vec![], 0),
        &BI2DE_CATALOG_ENTRY,
    );
    assert!(double.diagnostics.is_empty());
    assert_eq!(
        double.outputs[0].shape.known_dims(),
        Some(vec![Some(3), Some(1)])
    );
    assert!(matches!(
        double.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            ..
        })
    ));
    let single = super::super::inference::infer_bi2de(
        &call(input(NumericClass::Single, 1, 4), vec![], 0),
        &BI2DE_CATALOG_ENTRY,
    );
    assert!(matches!(
        single.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            ..
        })
    ));
    assert_eq!(single.outputs[0].storage, StorageFact::Dense);
    let narrow_integer = super::super::inference::infer_bi2de(
        &call(input(NumericClass::UInt8, 1, 3), vec![], 0),
        &BI2DE_CATALOG_ENTRY,
    );
    assert!(matches!(
        narrow_integer.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt8,
            ..
        })
    ));
    let integer = super::super::inference::infer_bi2de(
        &call(input(NumericClass::UInt8, 2, 9), vec![], 0),
        &BI2DE_CATALOG_ENTRY,
    );
    assert_eq!(
        integer.outputs[0].shape.known_dims(),
        Some(vec![Some(2), Some(1)])
    );
    assert!(matches!(integer.outputs[0].kind, ValueKindFact::Unknown));
    assert_eq!(integer.outputs[0].residency, ResidencyFact::Host);
}

#[test]
fn invalid_input_and_base_literals_are_diagnosed() {
    let invalid = super::super::inference::infer_bi2de(
        &call(
            ValueFact::proven(
                ValueKindFact::String,
                ShapeFact::Scalar,
                StorageFact::Scalar,
            ),
            vec![],
            0,
        ),
        &BI2DE_CATALOG_ENTRY,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-BI2DE-DIGITS"));
    let base = super::super::inference::infer_bi2de(
        &call(
            input(NumericClass::Double, 1, 3),
            vec![LiteralValue::Unknown, LiteralValue::Number(1.0)],
            1,
        ),
        &BI2DE_CATALOG_ENTRY,
    );
    assert!(base
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-BI2DE-BASE"));
    assert_eq!(BI2DE_CATALOG_ENTRY.bindings, &REQUIRED_DEFAULT_BINDING);
    assert_eq!(BI2DE_CATALOG_ENTRY.documentation.examples.len(), 6);
}

#[test]
fn three_argument_controls_keep_base_and_order_roles_distinct() {
    let digits = input(NumericClass::Double, 1, 3);
    for (literals, code, argument) in [
        (
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("left-msb".into()),
                LiteralValue::String("right-msb".into()),
            ],
            "RM-BI2DE-BASE",
            1,
        ),
        (
            vec![
                LiteralValue::Unknown,
                LiteralValue::Number(2.0),
                LiteralValue::Number(1.0),
            ],
            "RM-BI2DE-ORDER",
            2,
        ),
        (
            vec![
                LiteralValue::Unknown,
                LiteralValue::Number(4_503_599_627_370_497.0),
                LiteralValue::String("left-msb".into()),
            ],
            "RM-BI2DE-BASE",
            1,
        ),
        (
            vec![
                LiteralValue::Unknown,
                LiteralValue::Bool(true),
                LiteralValue::String("left-msb".into()),
            ],
            "RM-BI2DE-BASE",
            1,
        ),
    ] {
        let inferred = super::super::inference::infer_bi2de(
            &call(digits.clone(), literals, 2),
            &BI2DE_CATALOG_ENTRY,
        );
        assert!(
            inferred
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == code && diagnostic.argument == Some(argument)),
            "{inferred:?}"
        );
    }
}
