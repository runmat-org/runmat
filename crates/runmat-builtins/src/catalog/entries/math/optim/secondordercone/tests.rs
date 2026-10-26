use super::*;
use runmat_types::*;

#[test]
fn cone_descriptor_normalizes_double_fields_and_checks_dimensions() {
    let kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Int64,
        domain: NumericDomain::Real,
    });
    let mut request = CallRequest {
        arguments: vec![
            ValueFact::proven(
                kind.clone(),
                ShapeFact::from(vec![Some(2), Some(3)]),
                StorageFact::Dense,
            ),
            ValueFact::proven(
                kind.clone(),
                ShapeFact::from(vec![Some(1), Some(2)]),
                StorageFact::Dense,
            ),
            ValueFact::proven(
                kind.clone(),
                ShapeFact::from(vec![Some(3), Some(1)]),
                StorageFact::Dense,
            ),
            ValueFact::scalar(kind),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = inference::infer(&request, &SECONDORDERCONE_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    let ValueKindFact::Struct(fields) = &result.outputs[0].kind else {
        panic!("descriptor");
    };
    assert!(fields.fields_complete);
    assert_eq!(
        fields.fields["b"].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
    assert_eq!(
        fields.fields["A"].numeric().unwrap().class,
        NumericClass::Double
    );
    request.arguments[0].storage = StorageFact::Sparse;
    let sparse = inference::infer(&request, &SECONDORDERCONE_CATALOG_ENTRY);
    assert!(sparse.diagnostics.is_empty());
    let ValueKindFact::Struct(fields) = &sparse.outputs[0].kind else {
        panic!("descriptor")
    };
    assert_eq!(fields.fields["A"].storage, StorageFact::Sparse);
    assert_eq!(
        fields.fields["A"].numeric().unwrap().class,
        NumericClass::Double
    );
    request.arguments[2].shape = ShapeFact::from(vec![Some(2), Some(1)]);
    assert!(!inference::infer(&request, &SECONDORDERCONE_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
