use super::*;
use runmat_types::*;
fn request(input: ValueFact) -> CallRequest {
    CallRequest {
        arguments: vec![input],
        literals: LiteralContext::default(),
        outputs: OutputSelection::default(),
    }
}
#[test]
fn logical_host_masks_preserve_full_numeric_geometry_and_resident_scalar_policy() {
    for dims in [vec![], vec![Some(3)], vec![Some(2), Some(0), Some(3)]] {
        for class in [
            NumericClass::Single,
            NumericClass::Double,
            NumericClass::Int64,
            NumericClass::UInt64,
        ] {
            let input = ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(dims.clone()),
                StorageFact::Dense,
            );
            let result = inference::infer(&request(input.clone()), &ISHANDLE_CATALOG_ENTRY);
            assert!(result.diagnostics.is_empty());
            assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
            assert_eq!(result.outputs[0].shape, input.shape);
            assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
            assert_eq!(result.effects, ISHANDLE_CATALOG_ENTRY.contract.effect_set());
            assert_eq!(
                result.capabilities,
                ISHANDLE_CATALOG_ENTRY.contract.capability_set()
            );
            let mut resident = input;
            resident.residency = ResidencyFact::Device { provider: None };
            let expected = if resident.shape.element_count() == Some(1) {
                ShapeFact::Scalar
            } else {
                resident.shape.clone()
            };
            let result = inference::infer(&request(resident), &ISHANDLE_CATALOG_ENTRY);
            assert_eq!(result.outputs[0].shape, expected);
            assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
        }
    }
}
#[test]
fn nonnumeric_candidates_and_output_demand_follow_actual_contract() {
    let input = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    assert_eq!(
        inference::infer(&request(input.clone()), &ISHANDLE_CATALOG_ENTRY).outputs[0].shape,
        ShapeFact::Scalar
    );
    for count in [0, 1, 2] {
        let mut request = request(input.clone());
        request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(count));
        assert_eq!(
            inference::infer(&request, &ISHANDLE_CATALOG_ENTRY)
                .diagnostics
                .is_empty(),
            count <= 1
        );
    }
    let mut invalid = request(input);
    invalid.arguments.clear();
    assert!(!inference::infer(&invalid, &ISHANDLE_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
