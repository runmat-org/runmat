use super::*;

#[test]
fn string_addition_owns_converted_shape_type_and_host_placement() {
    use runmat_types::StorageFact;
    let text = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    let chars = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(2), Some(4)]),
        StorageFact::Dense,
    );
    for arguments in [vec![text.clone(), chars.clone()], vec![chars, text.clone()]] {
        let inferred = infer_catalog_call(&PLUS_CATALOG_ENTRY, &request(arguments));
        assert!(inferred.diagnostics.is_empty());
        assert_eq!(inferred.outputs[0].kind, ValueKindFact::String);
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
        assert_eq!(inferred.outputs[0].residency, ResidencyFact::Host);
        assert_eq!(inferred.outputs[0].storage, StorageFact::Dense);
    }
    let mut dynamic_text = text.clone();
    dynamic_text.certainty =
        runmat_types::CertaintyFact::Dynamic(runmat_types::DynamicReason::ExternalData);
    let inferred = infer_catalog_call(
        &PLUS_CATALOG_ENTRY,
        &request(vec![text.clone(), dynamic_text]),
    );
    assert_eq!(
        inferred.outputs[0].certainty,
        runmat_types::CertaintyFact::Dynamic(runmat_types::DynamicReason::ExternalData)
    );
    let mut resident = numeric(
        NumericClass::Single,
        NumericDomain::Real,
        &[Some(2), Some(1)],
    );
    resident.residency = ResidencyFact::Device { provider: None };
    assert!(
        !infer_catalog_call(&PLUS_CATALOG_ENTRY, &request(vec![text.clone(), resident]))
            .diagnostics
            .is_empty()
    );
    let mismatch = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    assert!(
        !infer_catalog_call(&PLUS_CATALOG_ENTRY, &request(vec![text, mismatch]))
            .diagnostics
            .is_empty()
    );
}
