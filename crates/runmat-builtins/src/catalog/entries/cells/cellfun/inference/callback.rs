use runmat_types::{
    CallRequest, CallableFact, CapabilitySet, DynamicReason, EffectSet, InferenceDiagnostic,
    LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount, ValueFact, ValueKindFact,
};

use super::super::shorthand::CellfunShorthand;

pub(super) struct Result {
    pub(super) output: Option<ValueFact>,
    pub(super) effects: EffectSet,
    pub(super) capabilities: CapabilitySet,
    pub(super) diagnostics: Vec<InferenceDiagnostic>,
}

pub(super) fn infer(
    request: &CallRequest,
    cell_indices: &[usize],
    extra_indices: &[usize],
) -> Result {
    let Some(callable) = request.arguments.first().and_then(callable) else {
        return Result {
            output: shorthand_output(request),
            effects: EffectSet::default(),
            capabilities: CapabilitySet::default(),
            diagnostics: Vec::new(),
        };
    };
    let mut arguments = cell_indices
        .iter()
        .filter_map(|index| request.arguments.get(*index))
        .map(cell_element)
        .collect::<Vec<_>>();
    arguments.extend(
        extra_indices
            .iter()
            .filter_map(|index| request.arguments.get(*index).cloned()),
    );
    let literals = LiteralContext::new(
        extra_indices
            .iter()
            .map(|index| {
                request
                    .literals
                    .literal_args
                    .get(*index)
                    .cloned()
                    .unwrap_or(LiteralValue::Unknown)
            })
            .collect(),
    );
    let inferred = crate::infer_callable_call(
        callable,
        &CallRequest {
            arguments,
            literals,
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
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

fn cell_element(value: &ValueFact) -> ValueFact {
    match &value.kind {
        ValueKindFact::Cell(cell) => (*cell.element).clone(),
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    }
}

fn shorthand_output(request: &CallRequest) -> Option<ValueFact> {
    let name = request
        .literals
        .literal_args
        .first()
        .and_then(crate::catalog::inference::literal_text)?;
    let shorthand = CellfunShorthand::parse(&name)?;
    if shorthand.returns_logical() {
        Some(ValueFact::scalar(ValueKindFact::Logical))
    } else {
        Some(ValueFact::scalar(ValueKindFact::Numeric(
            runmat_types::NumericFact {
                class: runmat_types::NumericClass::Double,
                domain: runmat_types::NumericDomain::Real,
            },
        )))
    }
}
