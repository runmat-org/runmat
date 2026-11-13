use super::super::super::super::super::infer_catalog_call;
use crate::{
    builtin_catalog_entry_by_name, BuiltinInferenceRule, MathInferenceRule, PowerOfTwoInferenceRule,
};
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn infer(input: ValueFact) -> runmat_types::CallInference {
    let entry = builtin_catalog_entry_by_name("nextpow2").expect("catalog entry");
    assert_eq!(
        entry.contract.inference_rule,
        BuiltinInferenceRule::Math(MathInferenceRule::PowerOfTwo(
            PowerOfTwoInferenceRule::NextExponent,
        ))
    );
    infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![input],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    )
}

#[test]
fn preserves_real_numeric_class_and_shape() {
    for class in [
        NumericClass::Single,
        NumericClass::Double,
        NumericClass::Int64,
        NumericClass::UInt64,
    ] {
        let inferred = infer(ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        ));
        assert!(inferred.diagnostics.is_empty(), "{class:?}");
        assert_eq!(
            inferred.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            })
        );
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
    }
}

#[test]
fn promotes_logical_and_rejects_unsupported_representations() {
    let logical = infer(ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::from(vec![Some(1), Some(4)]),
        StorageFact::Dense,
    ));
    assert!(logical.diagnostics.is_empty());
    assert_eq!(
        logical.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );

    for input in [
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        })),
        ValueFact::scalar(ValueKindFact::String),
    ] {
        assert!(!infer(input).diagnostics.is_empty());
    }
}
