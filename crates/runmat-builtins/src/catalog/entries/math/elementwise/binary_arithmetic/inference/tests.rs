use crate::{
    builtin_catalog_entry_by_name, infer_catalog_call, BinaryArithmeticInferenceRule,
    BuiltinInferenceRule, MathInferenceRule,
};
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

#[test]
fn plus_tracks_broadcast_class_domain_and_residency() {
    let entry = builtin_catalog_entry_by_name("plus").expect("plus catalog entry");
    assert_eq!(
        entry.contract.inference_rule,
        BuiltinInferenceRule::Math(MathInferenceRule::BinaryArithmetic(
            BinaryArithmeticInferenceRule::Add,
        ))
    );
    let mut left = numeric(
        NumericClass::Single,
        NumericDomain::Real,
        &[Some(3), Some(1)],
    );
    left.residency = ResidencyFact::Device {
        provider: Some("p".into()),
    };
    let right = numeric(
        NumericClass::Double,
        NumericDomain::Complex,
        &[Some(1), Some(4)],
    );
    let inferred = infer_catalog_call(entry, &request(vec![left, right]));
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(4)])
    );
    assert_eq!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex
        })
    );
    assert_eq!(
        inferred.outputs[0].residency,
        ResidencyFact::Device {
            provider: Some("p".into())
        }
    );
}

#[test]
fn plus_preserves_integer_and_applies_typed_like_placement() {
    let entry = builtin_catalog_entry_by_name("plus").expect("plus catalog entry");
    let integer = numeric(
        NumericClass::UInt64,
        NumericDomain::Real,
        &[Some(1), Some(2)],
    );
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut prototype = scalar.clone();
    prototype.residency = ResidencyFact::Device {
        provider: Some("gpu".into()),
    };
    let mut request = request(vec![
        integer,
        scalar,
        ValueFact::scalar(ValueKindFact::String),
        prototype,
    ]);
    request.literals.literal_args = vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::Keyword("like".into()),
        LiteralValue::Unknown,
    ];
    let inferred = infer_catalog_call(entry, &request);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].numeric().map(|fact| fact.class),
        Some(NumericClass::UInt64)
    );
    assert_eq!(
        inferred.outputs[0].residency,
        ResidencyFact::Device {
            provider: Some("gpu".into())
        }
    );
}

#[test]
fn plus_rejects_mixed_integer_classes_without_losing_the_broadcast_diagnostic() {
    let entry = builtin_catalog_entry_by_name("plus").expect("plus catalog entry");
    let inferred = infer_catalog_call(
        entry,
        &request(vec![
            numeric(
                NumericClass::Int16,
                NumericDomain::Real,
                &[Some(2), Some(2)],
            ),
            numeric(
                NumericClass::UInt16,
                NumericDomain::Real,
                &[Some(3), Some(2)],
            ),
        ]),
    );
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-BINARY-ARITHMETIC-INPUT"));
    assert_eq!(inferred.diagnostics.len(), 2);
}

fn numeric(class: NumericClass, domain: NumericDomain, shape: &[Option<usize>]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        ShapeFact::from(shape.to_vec()),
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
