use crate::catalog::inference::infer_catalog_call;
use crate::{
    builtin_catalog_entry_by_name, BuiltinInferenceRule, GammaFunctionInferenceRule,
    MathInferenceRule,
};
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn request(input: ValueFact) -> CallRequest {
    CallRequest {
        arguments: vec![input],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn gamma_preserves_float_class_and_shape_and_rejects_other_domains() {
    let entry = builtin_catalog_entry_by_name("gamma").expect("gamma catalog entry");
    assert_eq!(
        entry.contract.inference_rule,
        BuiltinInferenceRule::Math(MathInferenceRule::GammaFunction(
            GammaFunctionInferenceRule::Gamma,
        ))
    );
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let inferred = infer_catalog_call(entry, &request(input));
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

    for input in [
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int32,
            domain: NumericDomain::Real,
        })),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        })),
        ValueFact::scalar(ValueKindFact::Logical),
    ] {
        assert!(!infer_catalog_call(entry, &request(input))
            .diagnostics
            .is_empty());
    }
}

#[test]
fn gammaln_preserves_floating_facts_and_types_runmat_extensions() {
    let entry = builtin_catalog_entry_by_name("gammaln").expect("gammaln catalog entry");
    assert_eq!(
        entry.contract.inference_rule,
        BuiltinInferenceRule::Math(MathInferenceRule::GammaFunction(
            GammaFunctionInferenceRule::LogGamma,
        ))
    );
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let inferred = infer_catalog_call(entry, &request(input));
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

    for input in [
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        })),
        ValueFact::scalar(ValueKindFact::Logical),
        ValueFact::scalar(ValueKindFact::Character),
    ] {
        let inferred = infer_catalog_call(entry, &request(input));
        assert!(inferred.diagnostics.is_empty());
        assert_eq!(
            inferred.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })
        );
    }

    for input in [
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        })),
        ValueFact::scalar(ValueKindFact::String),
    ] {
        assert!(!infer_catalog_call(entry, &request(input))
            .diagnostics
            .is_empty());
    }
}
