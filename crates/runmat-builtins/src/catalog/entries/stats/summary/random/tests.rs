use super::*;
use runmat_types::*;
use std::collections::BTreeMap;

#[test]
fn random_literal_size_proves_shape_but_unknown_size_does_not() {
    let object = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::PROBABILITY_DISTRIBUTION.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let dimension = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let request = CallRequest {
        arguments: vec![object, dimension.clone(), dimension],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
            LiteralValue::Number(3.0),
        ]),
        outputs: OutputSelection::default(),
    };
    let result = infer(&request, &RANDOM_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    let mut unknown = request;
    unknown.literals = LiteralContext::default();
    assert_eq!(
        infer(&unknown, &RANDOM_CATALOG_ENTRY).outputs[0].shape,
        ShapeFact::Unknown
    );
}

#[test]
fn random_typed_integer_literal_size_is_not_rounded_through_double() {
    let object = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::PROBABILITY_DISTRIBUTION.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    let request = CallRequest {
        arguments: vec![object, scalar.clone(), scalar],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Integer {
                text: "9007199254740993".into(),
                class: NumericClass::UInt64,
            },
            LiteralValue::Integer {
                text: "0".into(),
                class: NumericClass::UInt64,
            },
        ]),
        outputs: OutputSelection::default(),
    };
    let inferred = infer(&request, &RANDOM_CATALOG_ENTRY);
    if let Ok(exact) = usize::try_from(9_007_199_254_740_993u64) {
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![Some(exact), Some(0)])
        );
    } else {
        assert!(inferred
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-RANDOM-SIZE"));
    }
    let mut overflow = request;
    overflow.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Integer {
            text: "4294967296".into(),
            class: NumericClass::UInt64,
        },
        LiteralValue::Integer {
            text: "4294967296".into(),
            class: NumericClass::UInt64,
        },
    ]);
    assert!(infer(&overflow, &RANDOM_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-RANDOM-SIZE"));
}

#[test]
fn random_named_logical_parameter_is_not_a_static_class_error() {
    let request = CallRequest {
        arguments: vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Character("Poisson".into()),
            LiteralValue::Unknown,
        ]),
        outputs: OutputSelection::default(),
    };
    let inferred = infer(&request, &RANDOM_CATALOG_ENTRY);
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(
        inferred.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );

    let mut logical_size = request;
    logical_size
        .arguments
        .push(ValueFact::scalar(ValueKindFact::Logical));
    logical_size
        .literals
        .literal_args
        .push(LiteralValue::Bool(true));
    assert!(infer(&logical_size, &RANDOM_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-RANDOM-SIZE"));
}
