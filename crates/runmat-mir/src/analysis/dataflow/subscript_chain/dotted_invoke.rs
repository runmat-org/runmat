use crate::{MirIndexComponent, MirIndexing};
use runmat_types::{DynamicReason, ValueFact, ValueKindFact};

use super::{index_selectors, simple_operand_fact};

pub(super) fn infer(
    member: &ValueFact,
    indexing: &MirIndexing,
    sequence_use: runmat_types::SequenceUse,
    facts: &[Option<ValueFact>],
) -> (
    runmat_types::FactInference,
    runmat_types::EffectSet,
    runmat_types::CapabilitySet,
) {
    let ValueKindFact::Callable(callable) = &member.kind else {
        return (
            runmat_types::infer_index(
                member,
                indexing.kind,
                &index_selectors(indexing, facts),
                indexing.result_context,
            ),
            Default::default(),
            Default::default(),
        );
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
    let fact = match inferred.outputs.as_slice() {
        [] => ValueFact::unknown(DynamicReason::DynamicDispatch),
        [fact] => fact.clone(),
        outputs => ValueFact::scalar(ValueKindFact::OutputList(runmat_types::OutputListFact {
            outputs: outputs.to_vec(),
            variadic: inferred.dynamic_outputs,
        })),
    };
    (
        runmat_types::FactInference {
            fact,
            diagnostics: inferred.diagnostics,
        },
        inferred.effects,
        inferred.capabilities,
    )
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
