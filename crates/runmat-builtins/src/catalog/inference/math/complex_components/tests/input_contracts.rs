use super::*;
use runmat_types::{
    AliasFact, CallRequest, DynamicReason, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact, ViewFact,
};

#[test]
fn numeric_component_contracts_encode_logical_character_and_rejection_boundaries() {
    let request = |argument: ValueFact| CallRequest {
        arguments: vec![argument],
        literals: Default::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let logical = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let conj = infer_catalog_call(
        builtin_catalog_entry_by_name("conj").expect("conj entry"),
        &request(logical.clone()),
    );
    assert!(conj.diagnostics.is_empty());
    assert_eq!(conj.outputs[0].kind, ValueKindFact::Logical);
    assert_eq!(conj.outputs[0].shape, logical.shape);

    for name in ["real", "imag"] {
        let inference = infer_catalog_call(
            builtin_catalog_entry_by_name(name).expect("component entry"),
            &request(logical.clone()),
        );
        assert!(inference.diagnostics.is_empty(), "{name}");
        assert_eq!(
            inference.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            "{name}"
        );
        assert_eq!(inference.outputs[0].shape, logical.shape, "{name}");
        assert_eq!(inference.outputs[0].view, ViewFact::Materialized, "{name}");
        assert_eq!(inference.outputs[0].alias, AliasFact::Unique, "{name}");
    }

    let mut resident_logical = logical.clone();
    resident_logical.residency = ResidencyFact::Device {
        provider: Some("single-only-provider".into()),
    };
    let converted = infer_catalog_call(
        builtin_catalog_entry_by_name("real").expect("real entry"),
        &request(resident_logical),
    );
    assert_eq!(
        converted.outputs[0].residency,
        ResidencyFact::Unknown,
        "a class-changing logical projection cannot promise resident double output"
    );

    let scalar = infer_catalog_call(
        builtin_catalog_entry_by_name("imag").expect("imag entry"),
        &request(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        }))),
    );
    assert_eq!(scalar.outputs[0].storage, StorageFact::Scalar);

    for name in ["conj", "real", "imag"] {
        let character = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), Some(5)]),
            StorageFact::Dense,
        );
        let inference = infer_catalog_call(
            builtin_catalog_entry_by_name(name).expect("component entry"),
            &request(character),
        );
        assert!(inference.diagnostics.is_empty(), "{name}");
        assert_eq!(
            inference.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            "{name}"
        );
    }

    let sparse = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(2)]),
        StorageFact::Sparse,
    );
    let rejected = infer_catalog_call(
        builtin_catalog_entry_by_name("real").expect("real entry"),
        &request(sparse),
    );
    assert!(rejected
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-NUMERIC-COMPONENT-SPARSE"));
    assert_eq!(rejected.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(
        rejected.outputs[0].certainty,
        runmat_types::CertaintyFact::Dynamic(DynamicReason::UnsupportedRepresentation)
    );

    let rejected = infer_catalog_call(
        builtin_catalog_entry_by_name("imag").expect("imag entry"),
        &request(ValueFact::scalar(ValueKindFact::String)),
    );
    assert!(rejected
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-NUMERIC-COMPONENT-INPUT"));

    let dynamic_input = ValueFact {
        shape: ShapeFact::from(vec![Some(7), Some(9)]),
        ..ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    let dynamic = infer_catalog_call(
        builtin_catalog_entry_by_name("conj").expect("conj entry"),
        &request(dynamic_input.clone()),
    );
    assert!(dynamic.diagnostics.is_empty());
    assert_eq!(dynamic.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(dynamic.outputs[0].shape, dynamic_input.shape);
}
