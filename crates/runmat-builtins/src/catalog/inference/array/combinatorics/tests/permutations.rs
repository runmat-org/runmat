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
            runmat_types::StructFact {
                fields: Default::default(),
                fields_complete: true,
            },
        ))]),
    );
    assert!(!inferred.diagnostics.is_empty());
}
