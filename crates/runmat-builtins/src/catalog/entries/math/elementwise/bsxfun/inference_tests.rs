use crate::{builtin_catalog_entry_by_name, infer_catalog_call};
use runmat_types::{
    BuiltinId, CallRequest, CallableFact, CallableIdentity, CapabilitySet, DimensionFact,
    LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn numeric(class: NumericClass, shape: Vec<Option<usize>>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape),
        StorageFact::Dense,
    )
}

fn builtin(name: &str) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: Some(CallableIdentity::Builtin(BuiltinId(name.into()))),
        capabilities: CapabilitySet::default(),
        parameters: Vec::new(),
        parameters_complete: false,
        outputs: Vec::new(),
        outputs_complete: false,
        variadic_inputs: true,
        variadic_outputs: true,
        captures: Vec::new(),
        captures_complete: true,
    }))
}

fn infer(function: ValueFact, left: ValueFact, right: ValueFact) -> runmat_types::CallInference {
    infer_catalog_call(
        builtin_catalog_entry_by_name("bsxfun").expect("bsxfun entry"),
        &CallRequest {
            arguments: vec![function, left, right],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    )
}

#[test]
fn builtin_callback_uses_its_argument_dependent_contract() {
    let result = infer(
        builtin("plus"),
        numeric(NumericClass::UInt64, vec![Some(2), Some(1)]),
        numeric(NumericClass::UInt64, vec![Some(1), Some(3)]),
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].numeric().map(|numeric| numeric.class),
        Some(NumericClass::UInt64)
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(2), DimensionFact::Known(3)]
        }
    );
    assert_eq!(result.outputs[0].storage, StorageFact::Dense);
}

#[test]
fn relational_callback_produces_logical_even_for_empty_output() {
    let result = infer(
        builtin("gt"),
        numeric(NumericClass::Double, vec![Some(0), Some(3)]),
        numeric(NumericClass::Double, vec![Some(1), Some(3)]),
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
    assert_eq!(result.outputs[0].shape.element_count(), Some(0));
}

#[test]
fn declared_user_callback_output_is_preserved_without_name_resolution() {
    let callback = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: Some(CallableIdentity::DynamicName(runmat_types::SymbolName(
            "gt".into(),
        ))),
        capabilities: CapabilitySet::default(),
        parameters: Vec::new(),
        parameters_complete: false,
        outputs: vec![ValueFact::scalar(ValueKindFact::Character)],
        outputs_complete: true,
        variadic_inputs: true,
        variadic_outputs: false,
        captures: Vec::new(),
        captures_complete: true,
    }));
    let result = infer(
        callback,
        numeric(NumericClass::Double, vec![Some(2), Some(1)]),
        numeric(NumericClass::Double, vec![Some(1), Some(2)]),
    );
    assert_eq!(result.outputs[0].kind, ValueKindFact::Character);
}
