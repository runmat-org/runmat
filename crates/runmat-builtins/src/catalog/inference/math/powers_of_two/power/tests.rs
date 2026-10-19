use super::super::super::super::super::infer_catalog_call;
use crate::{
    builtin_catalog_entry_by_name, BuiltinInferenceRule, MathInferenceRule, PowerOfTwoInferenceRule,
};
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn infer(arguments: Vec<ValueFact>) -> runmat_types::CallInference {
    let entry = builtin_catalog_entry_by_name("pow2").expect("catalog entry");
    assert_eq!(
        entry.contract.inference_rule,
        BuiltinInferenceRule::Math(MathInferenceRule::PowerOfTwo(
            PowerOfTwoInferenceRule::Power,
        ))
    );
    infer_catalog_call(
        entry,
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    )
}

#[test]
fn unary_models_floating_complex_and_extension_classes() {
    for (input, class, domain) in [
        (
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Complex,
            }),
            NumericClass::Single,
            NumericDomain::Complex,
        ),
        (
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            }),
            NumericClass::Double,
            NumericDomain::Real,
        ),
        (
            ValueKindFact::Logical,
            NumericClass::Double,
            NumericDomain::Real,
        ),
        (
            ValueKindFact::Character,
            NumericClass::Double,
            NumericDomain::Real,
        ),
    ] {
        let inferred = infer(vec![ValueFact::proven(
            input,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        )]);
        assert!(inferred.diagnostics.is_empty());
        assert_eq!(
            inferred.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact { class, domain })
        );
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
    }
}

#[test]
fn binary_models_broadcast_class_and_complexity() {
    let left = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(3), Some(1)]),
        StorageFact::Dense,
    );
    let right = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(4)]),
        StorageFact::Dense,
    );
    let inferred = infer(vec![left, right]);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        })
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(4)])
    );
}

#[test]
fn rejects_sparse_and_non_numeric_inputs() {
    let sparse = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(2)]),
        StorageFact::Sparse,
    );
    assert!(!infer(vec![sparse]).diagnostics.is_empty());
    assert!(!infer(vec![ValueFact::scalar(ValueKindFact::String)])
        .diagnostics
        .is_empty());
}
