use runmat_types::{
    CallRequest, CallableFact, CapabilitySet, DynamicReason, EffectSet, InferenceDiagnostic,
    LiteralContext, OutputSelection, RequestedOutputCount, ValueFact, ValueKindFact,
};

pub(super) struct CallbackResult {
    pub output: Option<ValueFact>,
    pub effects: EffectSet,
    pub capabilities: CapabilitySet,
    pub diagnostics: Vec<InferenceDiagnostic>,
}

pub(super) fn result(request: &CallRequest) -> CallbackResult {
    let Some(callable) = request.arguments.first().and_then(callable) else {
        return CallbackResult {
            output: None,
            effects: EffectSet::default(),
            capabilities: CapabilitySet::default(),
            diagnostics: Vec::new(),
        };
    };
    let Some(left) = request.arguments.get(1) else {
        return declared_result(callable);
    };
    let Some(right) = request.arguments.get(2) else {
        return declared_result(callable);
    };
    let callback_request = CallRequest {
        arguments: vec![scalarized(left), scalarized(right)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let mut inferred = crate::infer_callable_call(callable, &callback_request);
    for diagnostic in &mut inferred.diagnostics {
        diagnostic.argument = diagnostic.argument.map(|argument| argument + 1);
    }
    let output = inferred.outputs.into_iter().next();
    let mut diagnostics = inferred.diagnostics;
    if output
        .as_ref()
        .and_then(|output| output.shape.element_count())
        .is_some_and(|count| count != 1)
    {
        diagnostics.push(crate::catalog::inference::argument_error(
            "RM-CATALOG-BSXFUN-CALLBACK-OUTPUT",
            "bsxfun callbacks must return one scalar value per invocation",
            0,
        ));
    }
    CallbackResult {
        output,
        effects: inferred.effects,
        capabilities: inferred.capabilities,
        diagnostics,
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
    scalar
}

fn declared_result(callable: &CallableFact) -> CallbackResult {
    CallbackResult {
        output: callable
            .outputs
            .first()
            .cloned()
            .or_else(|| Some(ValueFact::unknown(DynamicReason::UnresolvedCallable))),
        effects: EffectSet::default(),
        capabilities: callable.capabilities.clone(),
        diagnostics: Vec::new(),
    }
}
