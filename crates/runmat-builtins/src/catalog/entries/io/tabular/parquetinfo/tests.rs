use super::*;
use runmat_types::*;

#[test]
fn metadata_has_exact_field_carriers_without_inventing_external_counts() {
    let request = CallRequest {
        arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        literals: LiteralContext::new(vec![LiteralValue::Unknown]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = inference::infer(&request, &PARQUETINFO_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    let ValueKindFact::Struct(info) = &result.outputs[0].kind else {
        panic!("scalar metadata struct");
    };
    assert!(info.fields_complete);
    assert_eq!(info.fields.len(), 12);
    assert_eq!(
        info.fields["NumRows"].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(
        info.fields["VariableNames"].shape,
        ShapeFact::from(vec![Some(1), None])
    );
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    let mut paged_text = request.clone();
    paged_text.arguments[0] = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), Some(1), Some(10)]),
        StorageFact::Dense,
    );
    assert!(inference::infer(&paged_text, &PARQUETINFO_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    paged_text.arguments[0].shape = ShapeFact::from(vec![Some(2), Some(1), Some(5)]);
    assert!(!inference::infer(&paged_text, &PARQUETINFO_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    let mut bad = request.clone();
    bad.arguments.clear();
    assert!(!inference::infer(&bad, &PARQUETINFO_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    bad = request;
    bad.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
    assert!(!inference::infer(&bad, &PARQUETINFO_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
