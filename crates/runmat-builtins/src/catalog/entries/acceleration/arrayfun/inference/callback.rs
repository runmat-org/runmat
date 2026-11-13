use runmat_types::{
    CallRequest, CallableFact, CapabilitySet, DynamicReason, EffectSet, InferenceDiagnostic,
    LiteralContext, OutputSelection, RequestedOutputCount, ValueFact, ValueKindFact,
};

pub(super) struct Result {
    pub output: Option<ValueFact>,
    pub effects: EffectSet,
    pub capabilities: CapabilitySet,
    pub diagnostics: Vec<InferenceDiagnostic>,
}

pub(super) fn infer(request: &CallRequest, array_indices: &[usize]) -> Result {
    let Some(callable) = request.arguments.first().and_then(callable) else {
        return Result {
            output: None,
            effects: EffectSet::default(),
            capabilities: CapabilitySet::default(),
            diagnostics: Vec::new(),
        };
    };
    let arguments = array_indices
        .iter()
        .filter_map(|index| request.arguments.get(*index))
        .map(scalarized)
        .collect();
    let callback_request = CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let mut inferred = crate::infer_callable_call(callable, &callback_request);
    for diagnostic in &mut inferred.diagnostics {
        diagnostic.argument = diagnostic.argument.map(|argument| argument + 1);
    }
    Result {
        output: inferred.outputs.into_iter().next(),
        effects: inferred.effects,
        capabilities: inferred.capabilities,
        diagnostics: inferred.diagnostics,
    }
}

fn callable(value: &ValueFact) -> Option<&CallableFact> {
    match &value.kind {
        ValueKindFact::Callable(callable) => Some(callable),
        _ => None,
    }
}

fn scalarized(value: &ValueFact) -> ValueFact {
    let mut scalar = ValueFact::scalar(value.kind.clone());
    scalar.residency = runmat_types::ResidencyFact::Host;
    scalar.certainty = match &value.certainty {
        runmat_types::CertaintyFact::Dynamic(reason) => {
            runmat_types::CertaintyFact::Dynamic(reason.clone())
        }
        certainty => certainty.clone(),
    };
    if matches!(scalar.kind, ValueKindFact::Unknown) {
        scalar.certainty = runmat_types::CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    scalar
}
