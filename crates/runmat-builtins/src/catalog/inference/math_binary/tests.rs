use std::collections::BTreeMap;

use super::super::infer_catalog_call;
use crate::{
    builtin_catalog_entry_by_name, BuiltinInferenceRule, MathInferenceRule, RemainderFunction,
};
use runmat_types::{
    standard, AliasFact, CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact,
    ObjectFact, OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact, ViewFact,
};

fn numeric(class: NumericClass, domain: NumericDomain, shape: Vec<Option<usize>>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        ShapeFact::from(shape),
        StorageFact::Dense,
    )
}

fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

fn object(class: runmat_types::StaticClassIdentity) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(class.owned()),
        properties: BTreeMap::from([(
            "Variables".into(),
            ValueFact::scalar(ValueKindFact::Unknown),
        )]),
        properties_complete: true,
        handle_semantics: None,
    }))
}

#[test]
fn remainder_rules_are_typed_and_preserve_exact_numeric_facts() {
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
        assert_eq!(
            output.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            }),
            "{name}"
        );
        assert_eq!(
            output.shape,
            ShapeFact::from(vec![Some(2), Some(3)]),
            "{name}"
        );
        assert_eq!(
            output.residency,
            ResidencyFact::Device {
                provider: Some("provider-a".into())
            },
            "{name}"
        );
        assert_eq!(output.view, ViewFact::Materialized, "{name}");
        assert_eq!(output.alias, AliasFact::Unique, "{name}");

        let mixed = infer_catalog_call(
            entry,
            &request(vec![
                numeric(
                    NumericClass::Single,
                    NumericDomain::Real,
                    vec![Some(2), Some(2)],
                ),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
            ]),
        );
        assert_eq!(
            mixed.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            }),
            "{name}"
        );
    }
}

#[test]
fn remainder_rules_reject_invalid_representations_and_output_counts() {
    for name in ["mod", "rem"] {
        let entry = builtin_catalog_entry_by_name(name).expect("remainder catalog entry");
        let cases = [
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
        ];
        for (left, code) in cases {
            let inferred = infer_catalog_call(
                entry,
                &request(vec![
                    left,
                    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Double,
                        domain: NumericDomain::Real,
                    })),
                ]),
            );
            assert!(
                inferred
                    .diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.code == code),
                "{name}: {code}"
            );
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
fn remainder_rules_preserve_supported_object_identity_without_stale_properties() {
    for name in ["mod", "rem"] {
        let entry = builtin_catalog_entry_by_name(name).expect("remainder catalog entry");
        for class in [standard::TABLE, standard::TIMETABLE, standard::DURATION] {
            let inferred = infer_catalog_call(
                entry,
                &request(vec![
                    object(class),
                    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Double,
                        domain: NumericDomain::Real,
                    })),
                ]),
            );
            assert!(inferred.diagnostics.is_empty(), "{name}: {class}");
            let ValueKindFact::Object(output) = &inferred.outputs[0].kind else {
                panic!("{name} must retain {class} identity");
            };
            assert_eq!(output.runtime_class, Some(class.owned()), "{name}");
            assert!(output.properties.is_empty(), "{name}");
            assert!(!output.properties_complete, "{name}");
        }

        let incompatible = infer_catalog_call(
            entry,
            &request(vec![object(standard::TABLE), object(standard::TIMETABLE)]),
        );
        assert!(incompatible
            .diagnostics
            .iter()
            .any(|diagnostic| { diagnostic.code == "RM-CATALOG-REMAINDER-OBJECT-PAIR" }));
    }
}
