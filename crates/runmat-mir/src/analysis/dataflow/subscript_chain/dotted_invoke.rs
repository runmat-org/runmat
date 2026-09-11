use crate::{MirIndexComponent, MirIndexing};
use runmat_types::{
    DynamicReason, SequenceFactInference, ValueFact, ValueKindFact, ValueSequenceFact,
};

use super::{index_selectors, simple_operand_fact};

/// Compatibility adapter for value-only inference callers. New sequence-aware
/// MIR analysis must use [`infer_sequence`] and retain the sequence fact.
pub(super) fn infer_legacy_value(
    member: &ValueFact,
    indexing: &MirIndexing,
    sequence_use: runmat_types::SequenceUse,
    facts: &[Option<ValueFact>],
) -> (
    runmat_types::FactInference,
    runmat_types::EffectSet,
    runmat_types::CapabilitySet,
) {
    let (inferred, effects, capabilities) = infer_sequence(member, indexing, sequence_use, facts);
    let fact = legacy_selected_fact(&inferred.sequence, sequence_use);
    (
        runmat_types::FactInference {
            fact,
            diagnostics: inferred.diagnostics,
        },
        effects,
        capabilities,
    )
}

pub(super) fn infer_sequence(
    member: &ValueFact,
    indexing: &MirIndexing,
    sequence_use: runmat_types::SequenceUse,
    facts: &[Option<ValueFact>],
) -> (
    SequenceFactInference,
    runmat_types::EffectSet,
    runmat_types::CapabilitySet,
) {
    let ValueKindFact::Callable(callable) = &member.kind else {
        let inference = runmat_types::infer_index_sequence(
            member,
            indexing.kind,
            &index_selectors(indexing, facts),
            indexing.result_context,
        );
        return (inference, Default::default(), Default::default());
    };
    let request = runmat_types::CallRequest {
        arguments: indexing
            .components
            .iter()
            .map(|component| match component {
                MirIndexComponent::Expr(operand) => simple_operand_fact(operand, facts),
                MirIndexComponent::Colon | MirIndexComponent::ContextualExpr(_) => {
                    ValueFact::unknown(DynamicReason::DynamicDispatch)
                }
            })
            .collect(),
        literals: Default::default(),
        outputs: runmat_types::OutputSelection::new(requested_outputs(sequence_use)),
    };
    let inferred = runmat_types::infer_call(
        &callable.call_contract(DynamicReason::DynamicDispatch),
        &request,
    );
    let sequence = ValueSequenceFact {
        outputs: inferred.outputs,
        variadic: inferred.dynamic_outputs,
    };
    (
        SequenceFactInference {
            sequence,
            diagnostics: inferred.diagnostics,
        },
        inferred.effects,
        inferred.capabilities,
    )
}

fn legacy_selected_fact(
    sequence: &ValueSequenceFact,
    sequence_use: runmat_types::SequenceUse,
) -> ValueFact {
    match sequence_use {
        runmat_types::SequenceUse::Discard => ValueFact::scalar(ValueKindFact::Void),
        runmat_types::SequenceUse::RequireSingle => {
            sequence.first_or_else(|| ValueFact::unknown(DynamicReason::DynamicDispatch))
        }
        _ => ValueFact::scalar(ValueKindFact::OutputList(runmat_types::OutputListFact {
            outputs: sequence.outputs.clone(),
            variadic: sequence.variadic,
        })),
    }
}

fn requested_outputs(use_: runmat_types::SequenceUse) -> runmat_types::RequestedOutputCount {
    match use_ {
        runmat_types::SequenceUse::Discard => runmat_types::RequestedOutputCount::Zero,
        runmat_types::SequenceUse::RequireSingle => runmat_types::RequestedOutputCount::One,
        runmat_types::SequenceUse::SelectPrefix { count } => {
            runmat_types::RequestedOutputCount::Exactly(count)
        }
        runmat_types::SequenceUse::SelectCurrentFunctionOutputs
        | runmat_types::SequenceUse::ExpandAll => {
            runmat_types::RequestedOutputCount::CurrentFunctionNargout
        }
        runmat_types::SequenceUse::SelectDestinationCardinality => {
            runmat_types::RequestedOutputCount::DestinationSequenceCardinality
        }
    }
}
