use crate::{
    builtin_catalog_entry_by_name, infer_catalog_call, BuiltinInferenceRule, MathInferenceRule,
    RemainderFunction,
};
use runmat_types::{
    standard, AliasFact, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    ViewFact,
};

use super::super::test_support::{numeric, object, request};

#[test]
fn typed_rules_preserve_numeric_facts() {
    for (name, function) in [
        ("mod", RemainderFunction::Modulus),
        ("rem", RemainderFunction::Remainder),
    ] {
        let entry = builtin_catalog_entry_by_name(name).expect("remainder catalog entry");
        assert_eq!(
            entry.contract.inference_rule,
            BuiltinInferenceRule::Math(MathInferenceRule::Remainder(function))
        );
        let mut left = numeric(
            NumericClass::UInt64,
            NumericDomain::Real,
            vec![Some(2), Some(1)],
        );
        left.residency = ResidencyFact::Device {
            provider: Some("provider-a".into()),
        };
        let mut right = numeric(
            NumericClass::UInt64,
            NumericDomain::Real,
            vec![Some(1), Some(3)],
        );
        right.residency = left.residency.clone();
        let inferred = infer_catalog_call(entry, &request(vec![left, right]));
        assert!(inferred.diagnostics.is_empty(), "{name}");
        let output = &inferred.outputs[0];
        assert!(matches!(
            output.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real
            })
        ));
        assert_eq!(output.shape, ShapeFact::from(vec![Some(2), Some(3)]));
        assert!(matches!(output.residency, ResidencyFact::Device { .. }));
        assert_eq!(output.view, ViewFact::Materialized);
        assert_eq!(output.alias, AliasFact::Unique);
    }
}

#[test]
fn rejects_invalid_numeric_representations_and_output_counts() {
    for name in ["mod", "rem"] {
        let entry = builtin_catalog_entry_by_name(name).expect("remainder catalog entry");
        for (left, code) in [
            (
                numeric(
                    NumericClass::Double,
                    NumericDomain::Complex,
                    vec![Some(2), Some(2)],
                ),
                "RM-CATALOG-REMAINDER-COMPLEX",
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
                "RM-CATALOG-REMAINDER-SPARSE",
            ),
        ] {
            let inferred = infer_catalog_call(
                entry,
                &request(vec![
                    left,
                    numeric(
                        NumericClass::Double,
                        NumericDomain::Real,
                        vec![Some(1), Some(1)],
                    ),
                ]),
            );
            assert!(inferred
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == code));
        }
        let incompatible = infer_catalog_call(
            entry,
            &request(vec![
                numeric(
                    NumericClass::Int32,
                    NumericDomain::Real,
                    vec![Some(2), Some(2)],
                ),
                numeric(
                    NumericClass::UInt32,
                    NumericDomain::Real,
                    vec![Some(2), Some(2)],
                ),
            ]),
        );
        assert!(incompatible
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-REMAINDER-INTEGER-CLASS"));
        let mut too_many = request(vec![
            ValueFact::scalar(ValueKindFact::Logical),
            ValueFact::scalar(ValueKindFact::Logical),
        ]);
        too_many.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
        assert!(!infer_catalog_call(entry, &too_many).diagnostics.is_empty());
    }
}

#[test]
fn preserves_supported_object_identity_without_stale_properties() {
    for name in ["mod", "rem"] {
        let entry = builtin_catalog_entry_by_name(name).expect("remainder catalog entry");
        for class in [standard::TABLE, standard::TIMETABLE, standard::DURATION] {
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
            assert!(inferred.diagnostics.is_empty(), "{name}: {class}");
            let ValueKindFact::Object(output) = &inferred.outputs[0].kind else {
                panic!("{name} must retain {class} identity");
            };
            assert_eq!(output.runtime_class, Some(class.owned()));
            assert!(output.properties.is_empty());
            assert!(!output.properties_complete);
        }
        for operands in [
            vec![object(standard::TABLE), object(standard::TIMETABLE)],
            vec![object(standard::DURATION), object(standard::TABLE)],
        ] {
            let inferred = infer_catalog_call(entry, &request(operands));
            assert!(!inferred.diagnostics.is_empty(), "{name}");
        }
    }
}
