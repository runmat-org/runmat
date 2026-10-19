use super::*;
use runmat_types::*;

#[test]
fn datatype_character_row_is_admitted_and_multiline_character_is_rejected() {
    let mut datatype = ValueFact::scalar(ValueKindFact::Character);
    datatype.shape = ShapeFact::from(vec![Some(1), Some(6)]);
    let request = |datatype: ValueFact| CallRequest {
        arguments: vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
            datatype,
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Number(1.0),
            LiteralValue::String("single".into()),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    assert!(
        inference::infer(&request(datatype.clone()), &WRITE_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
    datatype.shape = ShapeFact::from(vec![Some(2), Some(6)]);
    let invalid = inference::infer(&request(datatype), &WRITE_CATALOG_ENTRY);
    assert!(invalid
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-NET-WRITE-DATATYPE"));
}

fn request(arguments: Vec<ValueFact>, literal_args: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literal_args),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}

fn client() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(super::super::TCPCLIENT_CLASS.owned()),
        properties: Default::default(),
        properties_complete: false,
        handle_semantics: Some(true),
    }))
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
