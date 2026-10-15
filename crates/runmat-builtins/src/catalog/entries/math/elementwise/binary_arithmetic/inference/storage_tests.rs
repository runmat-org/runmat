use super::test_support::{numeric, request};
use crate::{
    infer_catalog_call, LDIVIDE_CATALOG_ENTRY, MINUS_CATALOG_ENTRY, PLUS_CATALOG_ENTRY,
    POWER_CATALOG_ENTRY, RDIVIDE_CATALOG_ENTRY, TIMES_CATALOG_ENTRY,
};
use runmat_types::{
    NumericClass, NumericDomain, NumericFact, ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

#[test]
fn sparse_storage_policy_is_selected_by_each_identity_contract() {
    let mut sparse = numeric(
        NumericClass::Double,
        NumericDomain::Real,
        &[Some(2), Some(3)],
    );
    sparse.storage = StorageFact::Sparse;
    sparse.residency = ResidencyFact::Host;
    for (entry, expected) in [
        (&PLUS_CATALOG_ENTRY, StorageFact::Sparse),
        (&MINUS_CATALOG_ENTRY, StorageFact::Sparse),
        (&TIMES_CATALOG_ENTRY, StorageFact::Sparse),
        (&RDIVIDE_CATALOG_ENTRY, StorageFact::Unknown),
        (&LDIVIDE_CATALOG_ENTRY, StorageFact::Unknown),
        (&POWER_CATALOG_ENTRY, StorageFact::Unknown),
    ] {
        let inferred = infer_catalog_call(entry, &request(vec![sparse.clone(), sparse.clone()]));
        assert_eq!(inferred.outputs[0].storage, expected);
        let mut dense = sparse.clone();
        dense.storage = StorageFact::Dense;
        for arguments in [
            vec![sparse.clone(), dense.clone()],
            vec![dense, sparse.clone()],
        ] {
            let inferred = infer_catalog_call(entry, &request(arguments));
            assert_eq!(inferred.outputs[0].storage, StorageFact::Unknown);
        }
    }
}

#[test]
fn preserving_sparse_requires_known_matching_operand_shapes() {
    let mut sparse = numeric(
        NumericClass::Double,
        NumericDomain::Real,
        &[Some(2), Some(3)],
    );
    sparse.storage = StorageFact::Sparse;
    sparse.residency = ResidencyFact::Host;
    for entry in [
        &PLUS_CATALOG_ENTRY,
        &MINUS_CATALOG_ENTRY,
        &TIMES_CATALOG_ENTRY,
    ] {
        for shape in [&[Some(1), Some(3)][..], &[None, Some(3)][..]] {
            let mut other = sparse.clone();
            other.shape = shape.to_vec().into();
            let inferred = infer_catalog_call(entry, &request(vec![sparse.clone(), other]));
            assert_eq!(inferred.outputs[0].storage, StorageFact::Unknown);
        }
    }
}

#[test]
fn sparse_storage_is_not_inferred_as_a_dense_or_scalar_buffer() {
    for entry in [
        &PLUS_CATALOG_ENTRY,
        &MINUS_CATALOG_ENTRY,
        &TIMES_CATALOG_ENTRY,
    ] {
        let name = entry.identity.name;
        for shape in [&[Some(1), Some(1)][..], &[Some(2), Some(3)][..]] {
            let mut sparse = numeric(NumericClass::Double, NumericDomain::Real, shape);
            sparse.storage = StorageFact::Sparse;
            let inferred =
                infer_catalog_call(entry, &request(vec![sparse.clone(), sparse.clone()]));
            assert_eq!(inferred.outputs[0].storage, StorageFact::Sparse, "{name}");
            let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }));
            let mixed = infer_catalog_call(entry, &request(vec![sparse, scalar]));
            assert_eq!(mixed.outputs[0].storage, StorageFact::Unknown, "{name}");
        }
        let mut unknown = numeric(NumericClass::Double, NumericDomain::Real, &[None, None]);
        unknown.storage = StorageFact::Sparse;
        let inferred = infer_catalog_call(entry, &request(vec![unknown.clone(), unknown]));
        assert_eq!(inferred.outputs[0].storage, StorageFact::Unknown, "{name}");
    }
}

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
