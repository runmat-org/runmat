use super::*;
use runmat_types::*;

fn unknown_client() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}

fn sealed_client() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(super::super::TCPCLIENT_CLASS.owned()),
        properties: Default::default(),
        properties_complete: false,
        handle_semantics: Some(true),
    }))
}

fn numeric_count() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

#[test]
fn datatype_character_row_is_admitted_and_multiline_character_is_rejected() {
    let mut datatype = ValueFact::scalar(ValueKindFact::Character);
    datatype.shape = ShapeFact::from(vec![Some(1), Some(6)]);
    let request = |datatype: ValueFact| CallRequest {
        arguments: vec![unknown_client(), numeric_count(), datatype],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
            LiteralValue::String("single".into()),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    assert!(
        inference::infer(&request(datatype.clone()), &READ_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
    datatype.shape = ShapeFact::from(vec![Some(2), Some(6)]);
    let invalid = inference::infer(&request(datatype), &READ_CATALOG_ENTRY);
    assert!(invalid
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-NET-READ-DATATYPE"));
}

#[test]
fn default_and_literal_datatype_keep_exact_host_output_class_and_count_shape() {
    let default = CallRequest {
        arguments: vec![sealed_client()],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let inferred = inference::infer(&default, &READ_CATALOG_ENTRY);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(inferred.outputs[0].residency, ResidencyFact::Host);
    assert_eq!(
        inferred.outputs[0].numeric().map(|numeric| numeric.class),
        Some(NumericClass::UInt8)
    );
    let single = CallRequest {
        arguments: vec![
            unknown_client(),
            numeric_count(),
            ValueFact::scalar(ValueKindFact::String),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Number(3.0),
            LiteralValue::String("SINGLE".into()),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let inferred = inference::infer(&single, &READ_CATALOG_ENTRY);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].numeric().map(|numeric| numeric.class),
        Some(NumericClass::Single)
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(1), DimensionFact::Known(3)]
        }
    );
}

#[test]
fn unknown_datatype_stays_dynamic_and_known_invalid_controls_diagnose() {
    let dynamic = CallRequest {
        arguments: vec![
            unknown_client(),
            numeric_count(),
            ValueFact::scalar(ValueKindFact::String),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let inferred = inference::infer(&dynamic, &READ_CATALOG_ENTRY);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(inferred.outputs[0].residency, ResidencyFact::Host);
    let invalid = CallRequest {
        arguments: vec![
            ValueFact::scalar(ValueKindFact::String),
            numeric_count(),
            ValueFact::scalar(ValueKindFact::String),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Number(0.0),
            LiteralValue::String("uint128".into()),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let inferred = inference::infer(&invalid, &READ_CATALOG_ENTRY);
    assert_eq!(inferred.diagnostics.len(), 3);
}
