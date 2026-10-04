use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, literal_args: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literal_args),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}

fn client() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}

fn double() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

#[test]
fn write_count_and_controls_have_sound_static_facts() {
    let valid = inference::infer(
        &request(
            vec![client(), double(), ValueFact::scalar(ValueKindFact::String)],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String("uint64".into()),
            ],
        ),
        &WRITE_CATALOG_ENTRY,
    );
    assert!(valid.diagnostics.is_empty());
    assert_eq!(valid.outputs.len(), 1);
    assert!(matches!(
        valid.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    ));
    assert_eq!(valid.outputs[0].shape, double().shape);

    let unknown_datatype = inference::infer(
        &request(
            vec![
                client(),
                double(),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::Unknown,
            ],
        ),
        &WRITE_CATALOG_ENTRY,
    );
    assert!(unknown_datatype.diagnostics.is_empty());

    let bad = inference::infer(
        &request(
            vec![client(), double(), ValueFact::scalar(ValueKindFact::String)],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String("uint128".into()),
            ],
        ),
        &WRITE_CATALOG_ENTRY,
    );
    assert!(bad
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-NET-WRITE-DATATYPE"));
    assert_eq!(WRITE_DESCRIPTOR.signatures.len(), 2);
    assert_eq!(WRITE_EXTENSIONS.len(), 4);
}
