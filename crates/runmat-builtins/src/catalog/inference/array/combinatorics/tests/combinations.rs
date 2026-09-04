use super::*;
use crate::{
    builtin_catalog_entry_by_name, ArrayInferenceRule, BuiltinInferenceRule,
    CombinatoricsInferenceRule,
};

#[test]
fn distinguishes_scalar_and_vector_results() {
    let entry = builtin_catalog_entry_by_name("nchoosek").expect("nchoosek entry");
    assert_eq!(
        entry.contract.inference_rule,
        BuiltinInferenceRule::Array(ArrayInferenceRule::Combinatorics(
            CombinatoricsInferenceRule::Combinations,
        ))
    );
    let selection = numeric(NumericClass::Double, ShapeFact::Scalar);
    let scalar = crate::infer_catalog_call(
        entry,
        &request(vec![
            numeric(NumericClass::Int16, ShapeFact::Scalar),
            selection.clone(),
        ]),
    );
    assert_eq!(scalar.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(
        scalar.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int16,
            domain: NumericDomain::Real,
        })
    );

    let vector = crate::infer_catalog_call(
        entry,
        &request(vec![
            numeric(
                NumericClass::Single,
                ShapeFact::from(vec![Some(1), Some(4)]),
            ),
            selection,
        ]),
    );
    assert_eq!(vector.outputs[0].shape, ShapeFact::Ranked { rank: 2 });
}

#[test]
fn coefficient_class_resolution_matches_runtime_rules() {
    let entry = builtin_catalog_entry_by_name("nchoosek").expect("nchoosek entry");
    let inferred = crate::infer_catalog_call(
        entry,
        &request(vec![
            numeric(NumericClass::Double, ShapeFact::Scalar),
            numeric(NumericClass::UInt16, ShapeFact::Scalar),
        ]),
    );
    assert_eq!(
        inferred.outputs[0].numeric().map(|numeric| numeric.class),
        Some(NumericClass::UInt16)
    );

    let mismatch = crate::infer_catalog_call(
        entry,
        &request(vec![
            numeric(NumericClass::UInt8, ShapeFact::Scalar),
            numeric(NumericClass::UInt16, ShapeFact::Scalar),
        ]),
    );
    assert!(mismatch.outputs[0].numeric().is_none());
    assert!(mismatch
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-NCHOOSEK-COEFFICIENT-CLASS"));
}

#[test]
fn rejects_missing_arguments() {
    let entry = builtin_catalog_entry_by_name("nchoosek").expect("nchoosek entry");
    let inferred = crate::infer_catalog_call(entry, &request(Vec::new()));
    assert!(!inferred.diagnostics.is_empty());
}
