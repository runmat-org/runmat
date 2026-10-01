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
