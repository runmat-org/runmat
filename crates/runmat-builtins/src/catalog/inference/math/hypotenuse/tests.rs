use crate::{
    builtin_catalog_entry_by_name, infer_catalog_call, BuiltinInferenceRule, MathInferenceRule,
};
use runmat_types::{
    standard, NumericClass, NumericDomain, NumericFact, OutputSelection, RequestedOutputCount,
    ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

use super::super::test_support::{numeric, object, request};

#[test]
fn tracks_real_output_broadcast_class_and_residency() {
    let entry = builtin_catalog_entry_by_name("hypot").expect("hypot catalog entry");
    assert_eq!(
        entry.contract.inference_rule,
        BuiltinInferenceRule::Math(MathInferenceRule::Hypot)
    );
    let mut left = numeric(
        NumericClass::Single,
        NumericDomain::Complex,
        vec![Some(2), Some(1)],
    );
    left.residency = ResidencyFact::Device {
        provider: Some("provider-a".into()),
    };
    let mut right = numeric(
        NumericClass::Single,
        NumericDomain::Real,
        vec![Some(1), Some(3)],
    );
    right.residency = left.residency.clone();
    let inferred = infer_catalog_call(entry, &request(vec![left, right]));
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    assert_eq!(
        inferred.outputs[0].residency,
        ResidencyFact::Device {
            provider: Some("provider-a".into())
        }
    );

    let mixed = infer_catalog_call(
        entry,
        &request(vec![
            numeric(
                NumericClass::Single,
                NumericDomain::Real,
                vec![Some(2), Some(2)],
            ),
            numeric(
                NumericClass::Double,
                NumericDomain::Real,
                vec![Some(2), Some(2)],
            ),
        ]),
    );
    assert_eq!(
        mixed.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );
}

#[test]
fn rejects_sparse_object_and_excess_outputs() {
    let entry = builtin_catalog_entry_by_name("hypot").expect("hypot catalog entry");
    let sparse = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(2)]),
        StorageFact::Sparse,
    );
    let inferred = infer_catalog_call(
        entry,
        &request(vec![
            sparse,
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
        ]),
    );
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-HYPOT-SPARSE"));

    let object_input = infer_catalog_call(
        entry,
        &request(vec![
            object(standard::TABLE),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
        ]),
    );
    assert!(object_input
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-HYPOT-INPUT"));

    let mut too_many = request(vec![
        ValueFact::scalar(ValueKindFact::Logical),
        ValueFact::scalar(ValueKindFact::Logical),
    ]);
    too_many.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
    assert!(!infer_catalog_call(entry, &too_many).diagnostics.is_empty());
}
