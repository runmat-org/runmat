use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn output_count_preserves_statement_and_gated_byte_result_shape() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let data = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let arguments = vec![text, data];
    let literals = vec![
        LiteralValue::String("out.csv".into()),
        LiteralValue::Unknown,
    ];
    let statement = inference::infer(
        &request(arguments.clone(), literals.clone(), 0),
        &CSVWRITE_CATALOG_ENTRY,
    );
    assert!(statement.diagnostics.is_empty());
    assert!(statement.outputs.is_empty());
    assert!(statement.effects.0.contains(&EffectKind::FilesystemWrite));
    let bytes = inference::infer(&request(arguments, literals, 1), &CSVWRITE_CATALOG_ENTRY);
    assert!(matches!(bytes.outputs[0].kind, ValueKindFact::Numeric(_)));
    assert_eq!(bytes.outputs[0].shape, ShapeFact::Scalar);
}

#[test]
fn malformed_known_controls_reject_without_guessing_dynamic_data() {
    let bad = inference::infer(
        &request(
            vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Character),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            ],
            vec![
                LiteralValue::String("a.csv".into()),
                LiteralValue::Unknown,
                LiteralValue::Number(f64::NAN),
                LiteralValue::Unknown,
            ],
            0,
        ),
        &CSVWRITE_CATALOG_ENTRY,
    );
    assert!(bad
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-CSVWRITE-DATA"));
    assert!(bad
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-CSVWRITE-OFFSET"));
}
