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

#[path = "storage_tests/publication.rs"]
mod publication;
