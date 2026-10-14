use super::*;

#[test]
fn unknown_population_shape_does_not_claim_vector_class() {
    let entry = crate::builtin_catalog_entry_by_name("nchoosek").unwrap();
    let output = crate::infer_catalog_call(
        entry,
        &request(vec![
            numeric(NumericClass::Double, ShapeFact::Unknown),
            numeric(NumericClass::UInt16, ShapeFact::Scalar),
        ]),
    );
    assert!(output.diagnostics.is_empty());
    assert!(matches!(output.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn literal_selection_infers_dense_vector_result_and_rejects_fraction() {
    let entry = crate::builtin_catalog_entry_by_name("nchoosek").unwrap();
    for (k, rows) in [(0.0, 1), (2.0, 6), (5.0, 0)] {
        let mut call = request(vec![
            numeric(
                NumericClass::Single,
                ShapeFact::from(vec![Some(1), Some(4)]),
            ),
            numeric(NumericClass::Double, ShapeFact::Scalar),
        ]);
        call.literals = LiteralContext::new(vec![
            runmat_types::LiteralValue::Unknown,
            runmat_types::LiteralValue::Number(k),
        ]);
        let output = crate::infer_catalog_call(entry, &call);
        assert!(output.diagnostics.is_empty());
        assert_eq!(
            output.outputs[0].shape,
            ShapeFact::from(vec![Some(rows), Some(k as usize)])
        );
        assert_eq!(output.outputs[0].storage, StorageFact::Dense);
    }
    let mut call = request(vec![
        numeric(NumericClass::Double, ShapeFact::Scalar),
        numeric(NumericClass::Double, ShapeFact::Scalar),
    ]);
    call.literals = LiteralContext::new(vec![
        runmat_types::LiteralValue::Number(5.0),
        runmat_types::LiteralValue::Real {
            text: "1.00000011920928955078125".into(),
            class: NumericClass::Single,
        },
    ]);
    assert!(crate::infer_catalog_call(entry, &call)
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-NCHOOSEK-SELECTION-VALUE"));
    call.literals.literal_args[1] = runmat_types::LiteralValue::Real {
        text: "1.00000001".into(),
        class: NumericClass::Single,
    };
    assert!(crate::infer_catalog_call(entry, &call)
        .diagnostics
        .is_empty());
}
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
            CombinatoricsInferenceRule::SelectionCombinations,
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
