use runmat_types::{
    CallRequest, DynamicReason, EffectSet, LiteralContext, LiteralValue, OutputSelection,
    SequenceFactInference, SequenceUse, ValueFact, ValueKindFact, ValueSequenceFact,
};

pub(super) fn infer(
    callable: &ValueFact,
    arguments: Vec<ValueFact>,
    literals: Vec<LiteralValue>,
    sequence_use: SequenceUse,
) -> Option<(
    SequenceFactInference,
    EffectSet,
    runmat_types::CapabilitySet,
)> {
    let ValueKindFact::Callable(callable) = &callable.kind else {
        return None;
    };
    let inferred = runmat_types::infer_call(
        &callable.call_contract(DynamicReason::DynamicDispatch),
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(requested_outputs(sequence_use)),
        },
    );
    Some((
        SequenceFactInference {
            sequence: ValueSequenceFact {
                outputs: inferred.outputs,
                variadic: inferred.dynamic_outputs,
            },
            diagnostics: inferred.diagnostics,
        },
        inferred.effects,
        inferred.capabilities,
    ))
}

pub(super) fn selected_fact(sequence: &ValueSequenceFact, sequence_use: SequenceUse) -> ValueFact {
    match sequence_use {
        SequenceUse::Discard => ValueFact::scalar(ValueKindFact::Void),
        SequenceUse::RequireSingle => {
            sequence.first_or_else(|| ValueFact::unknown(DynamicReason::DynamicDispatch))
        }
        SequenceUse::SelectPrefix { .. }
        | SequenceUse::SelectCurrentFunctionOutputs
        | SequenceUse::SelectDestinationCardinality
        | SequenceUse::ExpandAll => {
            ValueFact::scalar(ValueKindFact::OutputList(runmat_types::OutputListFact {
                outputs: sequence.outputs.clone(),
                variadic: sequence.variadic,
            }))
        }
    }
}

fn requested_outputs(use_: SequenceUse) -> runmat_types::RequestedOutputCount {
    match use_ {
        SequenceUse::Discard => runmat_types::RequestedOutputCount::Zero,
        SequenceUse::RequireSingle => runmat_types::RequestedOutputCount::One,
        SequenceUse::SelectPrefix { count } => runmat_types::RequestedOutputCount::Exactly(count),
        SequenceUse::SelectCurrentFunctionOutputs | SequenceUse::ExpandAll => {
            runmat_types::RequestedOutputCount::CurrentFunctionNargout
        }
        SequenceUse::SelectDestinationCardinality => {
            runmat_types::RequestedOutputCount::DestinationSequenceCardinality
        }
    }
}
