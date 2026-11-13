use super::*;

#[test]
fn abs_contract_preserves_class_shape_storage_and_residency_but_makes_complex_real() {
    use runmat_types::{
        CallRequest, NumericClass, NumericDomain, NumericFact, OutputSelection,
        RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    };
    let mut input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int64,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Sparse,
    );
    input.residency = runmat_types::ResidencyFact::Device {
        provider: Some("pilot".into()),
    };
    let request = CallRequest {
        arguments: vec![input.clone()],
        literals: runmat_types::LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let inference = infer_catalog_call(
        builtin_catalog_entry_by_name("abs").expect("abs entry"),
        &request,
    );
    assert!(inference.diagnostics.is_empty());
    assert_eq!(
        inference.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int64,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(inference.outputs[0].shape, input.shape);
    assert_eq!(inference.outputs[0].storage, input.storage);
    assert_eq!(inference.outputs[0].residency, input.residency);
}
