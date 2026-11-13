use super::*;
use runmat_types::*;

#[test]
fn cdf_named_upper_broadcasts_and_rejects_incompatible_geometry() {
    let family = ValueFact::scalar(ValueKindFact::String);
    let x = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Dense,
    );
    let mean = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    let sigma = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let upper = ValueFact::scalar(ValueKindFact::String);
    let request = CallRequest {
        arguments: vec![family, x, mean, sigma, upper],
        literals: LiteralContext::new(vec![
            LiteralValue::Character("Normal".into()),
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Character("upper".into()),
        ]),
        outputs: OutputSelection::default(),
    };
    let result = infer(&request, &CDF_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    let mut bad = request;
    bad.arguments[2].shape = ShapeFact::from(vec![Some(4), Some(3)]);
    assert!(infer(&bad, &CDF_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-DISTRIBUTION-SHAPE"));
}

#[test]
fn cdf_unknown_text_control_does_not_become_a_numeric_parameter() {
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let request = CallRequest {
        arguments: vec![
            ValueFact::scalar(ValueKindFact::String),
            scalar.clone(),
            scalar,
            ValueFact::scalar(ValueKindFact::String),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::String("Poisson".into()),
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
        ]),
        outputs: OutputSelection::default(),
    };
    let inference = infer(&request, &CDF_CATALOG_ENTRY);
    assert!(
        inference.diagnostics.is_empty(),
        "{:?}",
        inference.diagnostics
    );
    assert_eq!(inference.outputs[0].shape, ShapeFact::Unknown);
}
