use runmat_types::{
    CallRequest, CallableFact, CapabilitySet, DynamicReason, EffectSet, InferenceDiagnostic,
    LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount, ValueFact, ValueKindFact,
};

use super::super::shorthand::CellfunShorthand;
use crate::catalog::inference::gathered_value_fact;
use runmat_types::FactJoin;

pub(super) struct Result {
    pub(super) outputs: Vec<ValueFact>,
    pub(super) effects: EffectSet,
    pub(super) capabilities: CapabilitySet,
    pub(super) diagnostics: Vec<InferenceDiagnostic>,
}

pub(super) fn infer(
    request: &CallRequest,
    cell_indices: &[usize],
    extra_indices: &[usize],
    count: usize,
) -> Result {
    let mut result = primary(request, cell_indices, extra_indices, count);
    let start = super::options::Options::parse(request).start;
    for index in (start..request.arguments.len()).step_by(2) {
        let handler = request
            .literals
            .literal_args
            .get(index)
            .and_then(crate::catalog::inference::literal_text)
            .is_some_and(|name| name.trim().eq_ignore_ascii_case("ErrorHandler"));
        if !handler {
            continue;
        }
        let Some(callable) = request.arguments.get(index + 1).and_then(callable) else {
            result.outputs = vec![ValueFact::unknown(DynamicReason::UnresolvedCallable); count];
            continue;
        };
        let arguments = std::iter::once(ValueFact::unknown(DynamicReason::RuntimeValue))
            .chain(
                cell_indices
                    .iter()
                    .filter_map(|i| request.arguments.get(*i))
                    .map(cell_element),
            )
            .chain(
                extra_indices
                    .iter()
                    .filter_map(|i| request.arguments.get(*i))
                    .map(gathered_value_fact),
            )
            .collect();
        let inferred = crate::infer_callable_call(
            callable,
            &CallRequest {
                arguments,
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
            },
        );
        result.effects.0.extend(inferred.effects.0);
        result.capabilities.0.extend(inferred.capabilities.0);
        result.diagnostics.extend(inferred.diagnostics);
        for (output, recovery) in result.outputs.iter_mut().zip(inferred.outputs) {
            *output = output.join(&recovery);
        }
    }
    result
}

fn primary(
    request: &CallRequest,
    cell_indices: &[usize],
    extra_indices: &[usize],
    count: usize,
) -> Result {
    let Some(callable) = request.arguments.first().and_then(callable) else {
        return Result {
            outputs: vec![
                shorthand_output(request).unwrap_or_else(|| ValueFact::unknown(
                    DynamicReason::UnresolvedCallable
                ));
                count
            ],
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
            .filter_map(|index| request.arguments.get(*index))
            .map(gathered_value_fact),
    );
    let literals = LiteralContext::new(
        std::iter::repeat_n(LiteralValue::Unknown, cell_indices.len())
            .chain(extra_indices.iter().map(|index| {
                request
                    .literals
                    .literal_args
                    .get(*index)
                    .cloned()
                    .unwrap_or(LiteralValue::Unknown)
            }))
            .collect(),
    );
    let inferred = crate::infer_callable_call(
        callable,
        &CallRequest {
            arguments,
            literals,
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        },
    );
    Result {
        outputs: inferred.outputs,
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
        ValueKindFact::Cell(cell) => gathered_value_fact(&cell.element),
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

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{
        CellFact, NumericClass, NumericDomain, NumericFact, ResidencyFact, StorageFact,
    };

    #[test]
    fn callback_input_materialization_preserves_sparse_class() {
        let mut input = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }));
        input.residency = ResidencyFact::Device {
            provider: Some("test".into()),
        };
        input.storage = StorageFact::Sparse;
        let cell = ValueFact::scalar(ValueKindFact::Cell(CellFact {
            element: Box::new(input.clone()),
            elements: vec![input],
            elements_complete: true,
        }));
        let actual = cell_element(&cell);
        assert_eq!(actual.residency, ResidencyFact::Host);
        assert_eq!(actual.storage, StorageFact::Sparse);
        assert_eq!(actual.numeric().unwrap().class, NumericClass::UInt64);
    }
}
