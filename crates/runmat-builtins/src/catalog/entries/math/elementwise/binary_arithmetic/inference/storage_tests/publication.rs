use super::super::test_support::{numeric, request};
use crate::{
    infer_catalog_call, LDIVIDE_CATALOG_ENTRY, MINUS_CATALOG_ENTRY, PLUS_CATALOG_ENTRY,
    POWER_CATALOG_ENTRY, RDIVIDE_CATALOG_ENTRY, TIMES_CATALOG_ENTRY,
};
use runmat_types::{
    NumericClass, NumericDomain, NumericFact, ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

#[test]
fn floating_publication_carrier_does_not_promise_residency_or_power_domain() {
    for entry in [
        &PLUS_CATALOG_ENTRY,
        &MINUS_CATALOG_ENTRY,
        &TIMES_CATALOG_ENTRY,
        &RDIVIDE_CATALOG_ENTRY,
        &LDIVIDE_CATALOG_ENTRY,
        &POWER_CATALOG_ENTRY,
    ] {
        for (left_class, right_class) in [
            (NumericClass::Single, NumericClass::Single),
            (NumericClass::Single, NumericClass::Double),
            (NumericClass::Double, NumericClass::Single),
            (NumericClass::Double, NumericClass::Double),
        ] {
            let class =
                runmat_types::floating_binary_result_class(left_class, right_class).unwrap();
            for (left_shape, right_shape, count) in [
                (&[Some(1), Some(1)][..], &[Some(1), Some(1)][..], 1),
                (&[Some(2), Some(1)][..], &[Some(1), Some(3)][..], 6),
                (&[Some(0), Some(1)][..], &[Some(1), Some(3)][..], 0),
            ] {
                for residency in [
                    ResidencyFact::Host,
                    ResidencyFact::Unknown,
                    ResidencyFact::Device { provider: None },
                ] {
                    let mut left = numeric(left_class, NumericDomain::Real, left_shape);
                    let mut right = numeric(right_class, NumericDomain::Real, right_shape);
                    left.residency = residency.clone();
                    right.residency = residency;
                    let inferred = infer_catalog_call(entry, &request(vec![left, right]));
                    assert!(inferred.diagnostics.is_empty(), "{}", entry.identity.name);
                    let output = &inferred.outputs[0];
                    assert_eq!(output.shape.element_count(), Some(count));
                    assert_eq!(
                        output.storage,
                        if class == NumericClass::Double && count == 1 {
                            StorageFact::Unknown
                        } else {
                            StorageFact::Dense
                        },
                        "{}: {left_class:?}, {right_class:?}",
                        entry.identity.name
                    );
                    assert_eq!(output.residency, ResidencyFact::Unknown);
                    if entry.identity == POWER_CATALOG_ENTRY.identity {
                        assert_eq!(output.numeric(), None, "domain still needs admission");
                    }
                }
            }
        }
    }
}

#[test]
fn publication_carrier_requires_proven_inputs_broadcast_and_unmodified_output() {
    for entry in [
        &PLUS_CATALOG_ENTRY,
        &MINUS_CATALOG_ENTRY,
        &TIMES_CATALOG_ENTRY,
        &RDIVIDE_CATALOG_ENTRY,
        &LDIVIDE_CATALOG_ENTRY,
        &POWER_CATALOG_ENTRY,
    ] {
        let dense = numeric(
            NumericClass::Single,
            NumericDomain::Real,
            &[Some(2), Some(3)],
        );
        for storage in [
            StorageFact::Unknown,
            StorageFact::Opaque,
            StorageFact::Sparse,
        ] {
            let mut unproved = dense.clone();
            unproved.storage = storage;
            let inferred = infer_catalog_call(entry, &request(vec![dense.clone(), unproved]));
            assert_eq!(
                inferred.outputs[0].storage,
                StorageFact::Unknown,
                "{}: {storage:?}",
                entry.identity.name
            );
        }
        let mut unknown_shape = dense.clone();
        unknown_shape.shape = runmat_types::ShapeFact::Unknown;
        let inferred = infer_catalog_call(entry, &request(vec![dense.clone(), unknown_shape]));
        assert_eq!(inferred.outputs[0].storage, StorageFact::Unknown);
        let mut unknown_kind = dense.clone();
        unknown_kind.kind = ValueKindFact::Unknown;
        let inferred = infer_catalog_call(entry, &request(vec![dense.clone(), unknown_kind]));
        assert_eq!(inferred.outputs[0].storage, StorageFact::Unknown);
        let incompatible = numeric(
            NumericClass::Single,
            NumericDomain::Real,
            &[Some(4), Some(3)],
        );
        let inferred = infer_catalog_call(entry, &request(vec![dense.clone(), incompatible]));
        assert!(!inferred.diagnostics.is_empty());
        assert_eq!(inferred.outputs[0].storage, StorageFact::Unknown);

        let prototype = ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue);
        let keyword = ValueFact::scalar(ValueKindFact::String);
        let inferred = infer_catalog_call(
            entry,
            &request(vec![dense.clone(), dense, keyword, prototype]),
        );
        assert_eq!(inferred.outputs[0].storage, StorageFact::Unknown);

        // Genuine host scalar inputs retain the established scalar route;
        // geometry alone does not prove it for an unplaced tensor result.
        let mut scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }));
        scalar.residency = ResidencyFact::Host;
        let inferred = infer_catalog_call(entry, &request(vec![scalar.clone(), scalar.clone()]));
        assert_eq!(inferred.outputs[0].storage, StorageFact::Scalar);
        scalar.residency = ResidencyFact::Unknown;
        let inferred = infer_catalog_call(entry, &request(vec![scalar.clone(), scalar]));
        assert_eq!(inferred.outputs[0].storage, StorageFact::Unknown);
    }
}
