use super::*;
use crate::{
    builtin_catalog_entry_by_name, ArrayInferenceRule, BuiltinInferenceRule,
    CombinatoricsInferenceRule,
};

#[test]
fn preserves_class_and_infers_factorial_shape() {
    let entry = builtin_catalog_entry_by_name("perms").expect("perms entry");
    assert_eq!(
        entry.contract.inference_rule,
        BuiltinInferenceRule::Array(ArrayInferenceRule::Combinatorics(
            CombinatoricsInferenceRule::Permutations,
        ))
    );
    let inferred = crate::infer_catalog_call(
        entry,
        &request(vec![numeric(
            NumericClass::UInt64,
            ShapeFact::from(vec![Some(1), Some(3)]),
        )]),
    );
    assert_eq!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(6), Some(3)])
    );
}

#[test]
fn rejects_unsupported_representations() {
    let entry = builtin_catalog_entry_by_name("perms").expect("perms entry");
    let inferred = crate::infer_catalog_call(
        entry,
        &request(vec![ValueFact::scalar(ValueKindFact::Struct(
            runmat_types::StructFact::scalar(Default::default(), true),
        ))]),
    );
    assert!(!inferred.diagnostics.is_empty());
}

#[test]
fn rejects_proven_invalid_shapes_without_rejecting_unknown_dimensions() {
    let entry = builtin_catalog_entry_by_name("perms").unwrap();
    for shape in [
        vec![],
        vec![Some(2), Some(3)],
        vec![Some(0), Some(3)],
        vec![Some(1), Some(2), Some(1)],
    ] {
        let inferred = crate::infer_catalog_call(
            entry,
            &request(vec![numeric(NumericClass::Single, ShapeFact::from(shape))]),
        );
        assert!(inferred
            .diagnostics
            .iter()
            .any(|error| error.code == "RM-CATALOG-PERMS-INPUT"));
    }
    for shape in [
        ShapeFact::Unknown,
        ShapeFact::Ranked { rank: 2 },
        ShapeFact::from(vec![None, Some(1)]),
    ] {
        let inferred =
            crate::infer_catalog_call(entry, &request(vec![numeric(NumericClass::Single, shape)]));
        assert!(inferred.diagnostics.is_empty());
    }
    let oversized = crate::infer_catalog_call(
        entry,
        &request(vec![numeric(
            NumericClass::Double,
            ShapeFact::from(vec![Some(1), Some(11)]),
        )]),
    );
    assert!(oversized
        .diagnostics
        .iter()
        .any(|error| error.code == "RM-CATALOG-PERMS-SIZE"));
}

#[test]
fn scalar_identity_preserves_alias_facts() {
    let entry = builtin_catalog_entry_by_name("perms").unwrap();
    let mut input = ValueFact::scalar(ValueKindFact::String);
    input.alias = runmat_types::AliasFact::Shared;
    input.mutation = runmat_types::MutationFact::Immutable;
    let inferred = crate::infer_catalog_call(entry, &request(vec![input.clone()]));
    assert_eq!(inferred.outputs[0].alias, input.alias);
    assert_eq!(inferred.outputs[0].mutation, input.mutation);
}

#[test]
fn one_element_arrays_remain_dense_carriers() {
    let entry = builtin_catalog_entry_by_name("perms").unwrap();
    for kind in [
        ValueKindFact::Logical,
        ValueKindFact::String,
        ValueKindFact::Character,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        }),
    ] {
        let input = ValueFact::proven(
            kind,
            ShapeFact::from(vec![Some(1), Some(1)]),
            StorageFact::Dense,
        );
        let output = crate::infer_catalog_call(entry, &request(vec![input]));
        assert_eq!(output.outputs[0].storage, StorageFact::Dense);
    }
}
