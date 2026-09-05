use crate::{
    builtin_catalog_entry_by_name, infer_catalog_call, BuiltinInferenceRule, MathInferenceRule,
};
use runmat_types::{
    standard, AliasFact, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    ViewFact,
};

use super::super::test_support::{numeric, object, request};

#[test]
fn tracks_broadcast_class_residency_and_materialization() {
    let entry = builtin_catalog_entry_by_name("atan2").expect("atan2 catalog entry");
    assert_eq!(
        entry.contract.inference_rule,
        BuiltinInferenceRule::Math(MathInferenceRule::Atan2)
    );
    let mut y = numeric(
        NumericClass::UInt64,
        NumericDomain::Real,
        vec![Some(2), Some(1)],
    );
    y.residency = ResidencyFact::Device {
        provider: Some("provider-a".into()),
    };
    let inferred = infer_catalog_call(
        entry,
        &request(vec![
            y,
            numeric(
                NumericClass::Double,
                NumericDomain::Real,
                vec![Some(1), Some(3)],
            ),
        ]),
    );
    assert!(inferred.diagnostics.is_empty());
    let output = &inferred.outputs[0];
    assert_eq!(
        output.kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(output.shape, ShapeFact::from(vec![Some(2), Some(3)]));
    assert!(matches!(output.residency, ResidencyFact::Device { .. }));
    assert_eq!(output.view, ViewFact::Materialized);
    assert_eq!(output.alias, AliasFact::Unique);

    let single = infer_catalog_call(
        entry,
        &request(vec![
            numeric(
                NumericClass::UInt16,
                NumericDomain::Real,
                vec![Some(2), Some(2)],
            ),
            numeric(
                NumericClass::Single,
                NumericDomain::Real,
                vec![Some(2), Some(2)],
            ),
        ]),
    );
    assert!(matches!(
        single.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real
        })
    ));
}

#[test]
fn rejects_complex_sparse_and_excess_outputs() {
    let entry = builtin_catalog_entry_by_name("atan2").expect("atan2 catalog entry");
    for (operand, code) in [
        (
            numeric(
                NumericClass::Double,
                NumericDomain::Complex,
                vec![Some(2), Some(2)],
            ),
            "RM-CATALOG-ATAN2-COMPLEX",
        ),
        (
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(2), Some(2)]),
                StorageFact::Sparse,
            ),
            "RM-CATALOG-ATAN2-SPARSE",
        ),
    ] {
        let inferred = infer_catalog_call(
            entry,
            &request(vec![
                operand,
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
            ]),
        );
        assert!(inferred
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == code));
    }
    let mut too_many = request(vec![
        numeric(
            NumericClass::Double,
            NumericDomain::Real,
            vec![Some(1), Some(1)],
        ),
        numeric(
            NumericClass::Double,
            NumericDomain::Real,
            vec![Some(1), Some(1)],
        ),
    ]);
    too_many.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
    assert!(!infer_catalog_call(entry, &too_many).diagnostics.is_empty());
}

#[test]
fn preserves_matching_tabular_identity() {
    let entry = builtin_catalog_entry_by_name("atan2").expect("atan2 catalog entry");
    for class in [standard::TABLE, standard::TIMETABLE] {
        let inferred = infer_catalog_call(
            entry,
            &request(vec![
                object(class),
                numeric(
                    NumericClass::Double,
                    NumericDomain::Real,
                    vec![Some(1), Some(1)],
                ),
            ]),
        );
        assert!(inferred.diagnostics.is_empty(), "{class}");
        let ValueKindFact::Object(output) = &inferred.outputs[0].kind else {
            panic!("atan2 must retain {class} identity");
        };
        assert_eq!(output.runtime_class, Some(class.owned()));
        assert!(output.properties.is_empty());
        assert!(!output.properties_complete);
    }
    let incompatible = infer_catalog_call(
        entry,
        &request(vec![object(standard::TABLE), object(standard::TIMETABLE)]),
    );
    assert!(incompatible
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-TABULAR-BINARY-PAIR"));
}
