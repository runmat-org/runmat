use super::super::super::super::infer_catalog_call;
use crate::{builtin_catalog_entry_by_name, BuiltinInferenceRule, MathInferenceRule};
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn infer(input: ValueFact) -> runmat_types::CallInference {
    let entry = builtin_catalog_entry_by_name("heaviside").expect("heaviside catalog entry");
    assert_eq!(
        entry.contract.inference_rule,
        BuiltinInferenceRule::Math(MathInferenceRule::Heaviside)
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
fn preserves_floating_class_and_shape_and_promotes_extensions() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let result = infer(input);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(
        result.outputs[0].shape,
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
        let result = infer(input);
        assert!(result.diagnostics.is_empty());
        assert_eq!(
            result.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })
        );
    }
}

#[test]
fn preserves_symbolic_and_rejects_unsupported_representations() {
    let symbolic = infer(ValueFact::scalar(ValueKindFact::Symbolic));
    assert!(symbolic.diagnostics.is_empty());
    assert_eq!(symbolic.outputs[0].kind, ValueKindFact::Symbolic);

    for input in [
        ValueFact::scalar(ValueKindFact::String),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        })),
    ] {
        let result = infer(input);
        assert!(!result.diagnostics.is_empty());
    }
}
