use crate::{
    builtin_catalog_entry_by_name, infer_catalog_call, BuiltinDocumentationAuthority,
    BuiltinExampleVerification, BuiltinInferenceRule, MathInferenceRule,
};
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

#[test]
fn documentation_is_catalog_owned_and_executable() {
    for (name, example_count, faq_count) in [("double", 8, 9), ("single", 6, 10)] {
        let entry = builtin_catalog_entry_by_name(name).expect("floating conversion entry");
        let documentation = &entry.documentation;
        assert_eq!(
            documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(documentation.examples.len(), example_count, "{name}");
        assert_eq!(documentation.faqs.len(), faq_count, "{name}");
        assert!(
            documentation
                .sections
                .iter()
                .any(|section| section.heading == "GPU execution"),
            "{name}"
        );
        assert!(!documentation.evidence.implementation.is_empty(), "{name}");
        assert!(!documentation.evidence.verification.is_empty(), "{name}");
        assert!(documentation.examples.iter().all(|example| {
            !example.id.is_empty()
                && matches!(
                    example.verification,
                    BuiltinExampleVerification::Assertions { .. }
                )
        }));
    }
}

#[test]
fn contracts_type_class_shape_and_like_residency_separately() {
    let mut source = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    source.residency = ResidencyFact::Device {
        provider: Some("source-provider".into()),
    };
    let mut prototype = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Real,
    }));
    prototype.residency = ResidencyFact::Device {
        provider: Some("prototype-provider".into()),
    };

    for (name, class) in [
        ("double", NumericClass::Double),
        ("single", NumericClass::Single),
    ] {
        let entry = builtin_catalog_entry_by_name(name).expect("floating conversion entry");
        assert_eq!(
            entry.contract.inference_rule,
            BuiltinInferenceRule::Math(MathInferenceRule::NumericConversionWithLike(class))
        );

        let default_conversion = infer_catalog_call(
            entry,
            &CallRequest {
                arguments: vec![source.clone()],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
        );
        assert!(default_conversion.diagnostics.is_empty(), "{name}");
        assert_eq!(
            default_conversion.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Complex,
            }),
            "{name}"
        );
        assert_eq!(default_conversion.outputs[0].shape, source.shape, "{name}");
        assert_eq!(
            default_conversion.outputs[0].residency,
            ResidencyFact::Unknown,
            "provider capability decides default resident conversion"
        );

        let like_conversion = infer_catalog_call(
            entry,
            &CallRequest {
                arguments: vec![
                    source.clone(),
                    ValueFact::scalar(ValueKindFact::String),
                    prototype.clone(),
                ],
                literals: LiteralContext::new(vec![
                    LiteralValue::Unknown,
                    LiteralValue::String("like".into()),
                    LiteralValue::Unknown,
                ]),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
        );
        assert!(like_conversion.diagnostics.is_empty(), "{name}");
        assert_eq!(
            like_conversion.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Complex,
            }),
            "the prototype must not change {name}'s fixed result class"
        );
        assert_eq!(
            like_conversion.outputs[0].residency, prototype.residency,
            "{name}"
        );
    }

    let invalid = infer_catalog_call(
        builtin_catalog_entry_by_name("double").expect("double entry"),
        &CallRequest {
            arguments: vec![
                source,
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("like".into()),
                LiteralValue::Unknown,
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-NUMERIC-CONVERSION-PROTOTYPE"));

    let complex_prototype = infer_catalog_call(
        builtin_catalog_entry_by_name("single").expect("single entry"),
        &CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Single,
                    domain: NumericDomain::Complex,
                })),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("like".into()),
                LiteralValue::Unknown,
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(complex_prototype
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-NUMERIC-CONVERSION-PROTOTYPE"));
}
