use runmat_types::*;

pub(super) fn scalar(class: NumericClass, domain: NumericDomain) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact { class, domain }))
}

pub(super) fn request(
    arguments: Vec<ValueFact>,
    literals: Vec<LiteralValue>,
    outputs: usize,
) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

pub(super) fn callable_with_output(output: ValueFact) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: CapabilitySet::default(),
        parameters: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        parameters_complete: true,
        outputs: vec![output],
        outputs_complete: true,
        variadic_inputs: false,
        variadic_outputs: false,
        captures: vec![],
        captures_complete: true,
    }))
}
