use super::*;

#[test]
fn logarithm_contracts_preserve_shape_class_and_value_dependent_domain() {
    use runmat_types::{
        CallRequest, CertaintyFact, DynamicReason, LiteralContext, LiteralValue, NumericClass,
        NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact,
        ValueFact, ValueKindFact,
    };

    for (name, base) in [
        ("log", LogarithmKind::Natural),
        ("log10", LogarithmKind::Common),
    ] {
        let entry = builtin_catalog_entry_by_name(name).expect("logarithm entry");
        assert!(matches!(
            entry.contract.inference_rule,
            BuiltinInferenceRule::Math(MathInferenceRule::Logarithm(actual)) if actual == base
        ));
        let infer = |input: ValueFact, literal: LiteralValue| {
            infer_catalog_call(
                entry,
                &CallRequest {
                    arguments: vec![input],
                    literals: LiteralContext::new(vec![literal]),
                    outputs: OutputSelection::new(RequestedOutputCount::One),
                },
            )
        };

        let single = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        let negative = infer(single.clone(), LiteralValue::Number(-1.0));
        assert!(negative.diagnostics.is_empty(), "{name}");
        assert_eq!(
            negative.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Complex,
            }),
            "{name}"
        );
        assert_eq!(
            negative.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)]),
            "{name}"
        );

        let nonnegative = infer(single, LiteralValue::Number(1.0));
        assert_eq!(
            nonnegative.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            }),
            "{name}"
        );

        let unsigned = infer(
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            })),
            LiteralValue::Unknown,
        );
        assert_eq!(
            unsigned.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            "{name}"
        );

        let signed = infer(
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Int64,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(4), Some(2)]),
                StorageFact::Dense,
            ),
            LiteralValue::Unknown,
        );
        assert_eq!(signed.outputs[0].kind, ValueKindFact::Unknown, "{name}");
        assert_eq!(
            signed.outputs[0].shape,
            ShapeFact::from(vec![Some(4), Some(2)]),
            "{name}"
        );
        assert_eq!(
            signed.outputs[0].certainty,
            CertaintyFact::Dynamic(DynamicReason::RuntimeValue),
            "{name}"
        );
    }
}

#[test]
fn logarithm_contracts_cover_tabular_symbolic_and_rejected_inputs() {
    use std::collections::BTreeMap;

    use runmat_types::{
        CallRequest, ClassIdentity, LiteralContext, NumericClass, NumericDomain, NumericFact,
        ObjectFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact,
        ValueKindFact,
    };

    let request_for = |input| CallRequest {
        arguments: vec![input],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    for name in ["log", "log10"] {
        let entry = builtin_catalog_entry_by_name(name).expect("logarithm entry");
        for class in [
            runmat_types::standard::TABLE,
            runmat_types::standard::TIMETABLE,
        ] {
            let table = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(class.owned()),
                properties: BTreeMap::from([(
                    "Variables".into(),
                    ValueFact::scalar(ValueKindFact::Unknown),
                )]),
                properties_complete: true,
                handle_semantics: None,
            }));
            let inferred = infer_catalog_call(entry, &request_for(table));
            let ValueKindFact::Object(output) = &inferred.outputs[0].kind else {
                panic!("{name} must preserve tabular identity");
            };
            assert_eq!(output.runtime_class, Some(class.owned()), "{name}");
            assert!(output.properties.is_empty(), "{name}");
            assert!(!output.properties_complete, "{name}");
        }

        let complex_integer = infer_catalog_call(
            entry,
            &request_for(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Int32,
                domain: NumericDomain::Complex,
            }))),
        );
        assert!(complex_integer
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOGARITHM-COMPLEX-INTEGER"));

        let sparse = infer_catalog_call(
            entry,
            &request_for(ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(3), Some(3)]),
                StorageFact::Sparse,
            )),
        );
        assert!(sparse
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOGARITHM-SPARSE"));

        let too_many = infer_catalog_call(
            entry,
            &CallRequest {
                arguments: vec![
                    ValueFact::scalar(ValueKindFact::Logical),
                    ValueFact::scalar(ValueKindFact::Logical),
                ],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
        );
        assert!(too_many
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOGARITHM-ARITY"));
    }

    let symbolic = ValueFact::scalar(ValueKindFact::Symbolic);
    let log = infer_catalog_call(
        builtin_catalog_entry_by_name("log").expect("log entry"),
        &request_for(symbolic.clone()),
    );
    assert_eq!(log.outputs[0].kind, ValueKindFact::Symbolic);
    let log10 = infer_catalog_call(
        builtin_catalog_entry_by_name("log10").expect("log10 entry"),
        &request_for(symbolic),
    );
    assert!(log10
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOGARITHM-INPUT"));

    let user_object = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(ClassIdentity::new("UserClass").unwrap()),
        properties: BTreeMap::new(),
        properties_complete: false,
        handle_semantics: None,
    }));
    let overloaded = infer_catalog_call(
        builtin_catalog_entry_by_name("log").expect("log entry"),
        &request_for(user_object),
    );
    assert_eq!(overloaded.outputs[0].kind, ValueKindFact::Unknown);
}
