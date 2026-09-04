use super::*;
use runmat_types::{
    CallableFact, CapabilitySet, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact,
    ValueKindFact,
};

fn callable(outputs: Vec<ValueFact>) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: CapabilitySet::default(),
        parameters: Vec::new(),
        parameters_complete: false,
        outputs,
        outputs_complete: true,
        variadic_inputs: true,
        variadic_outputs: false,
        captures: Vec::new(),
        captures_complete: true,
    }))
}

#[test]
fn callback_outputs_keep_kind_and_trailing_shape_while_group_height_is_dynamic() {
    let integer = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt16,
        domain: NumericDomain::Real,
    }));
    let text = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    let data = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(8), Some(1)]),
        StorageFact::Dense,
    );
    let groups = data.clone();
    let inferred = infer(
        "splitapply",
        vec![callable(vec![integer, text]), data, groups],
        RequestedOutputCount::Exactly(2),
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![None, Some(1)])
    );
    assert_eq!(
        inferred.outputs[0].numeric().map(|numeric| numeric.class),
        Some(NumericClass::UInt16)
    );
    assert_eq!(
        inferred.outputs[1].shape,
        ShapeFact::from(vec![None, Some(3)])
    );
    assert!(matches!(inferred.outputs[1].kind, ValueKindFact::String));
}

#[test]
fn noncallable_and_short_calls_report_owned_diagnostics() {
    let inferred = infer(
        "splitapply",
        vec![ValueFact::scalar(ValueKindFact::String)],
        RequestedOutputCount::One,
    );
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-SPLITAPPLY-ARITY"));
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-SPLITAPPLY-FUNCTION"));
}
