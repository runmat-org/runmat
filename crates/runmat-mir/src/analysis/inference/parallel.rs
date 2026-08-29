use runmat_types::{DynamicReason, ValueFact, ValueKindFact};

use crate::analysis::engine::FlowState;

use super::operand_fact;

pub(crate) fn distributed_fact(
    operation: &crate::parallel::MirDistributedOp,
    state: &mut FlowState,
) -> ValueFact {
    use crate::parallel::MirDistributedOp;
    match operation {
        MirDistributedOp::Create {
            id,
            owner,
            input,
            scheme,
        } => {
            let value = operand_fact(input, state);
            state.distributed.insert(*id, value.clone());
            ValueFact::scalar(ValueKindFact::Distributed(runmat_types::DistributedFact {
                id: *id,
                owner: *owner,
                scheme: Some(scheme.clone()),
                value: Box::new(value),
                materializable: true,
            }))
        }
        MirDistributedOp::LocalPart { value } | MirDistributedOp::Materialize { value } => {
            match operand_fact(value, state).kind {
                ValueKindFact::Distributed(distributed) => *distributed.value,
                _ => dynamic_value(),
            }
        }
        MirDistributedOp::Redistribute { value, scheme } => {
            let ValueKindFact::Distributed(distributed) = operand_fact(value, state).kind else {
                return dynamic_value();
            };
            ValueFact::scalar(ValueKindFact::Distributed(runmat_types::DistributedFact {
                id: distributed.id,
                owner: distributed.owner,
                scheme: Some(scheme.clone()),
                value: distributed.value,
                materializable: distributed.materializable,
            }))
        }
    }
}

pub(crate) fn collective_fact(
    operation: &crate::parallel::MirCollectiveOp,
    state: &FlowState,
) -> ValueFact {
    use crate::parallel::MirCollectiveOp;
    match operation {
        MirCollectiveOp::Barrier { .. } | MirCollectiveOp::Send { .. } => {
            ValueFact::scalar(ValueKindFact::Void)
        }
        MirCollectiveOp::Broadcast { input, .. } => input.as_ref().map_or_else(
            || ValueFact::unknown(DynamicReason::RuntimeValue),
            |input| operand_fact(input, state),
        ),
        MirCollectiveOp::Gather { input, .. }
        | MirCollectiveOp::Scatter { input, .. }
        | MirCollectiveOp::AllGather { input, .. }
        | MirCollectiveOp::Reduce { input, .. }
        | MirCollectiveOp::AllReduce { input, .. }
        | MirCollectiveOp::SendReceive { input, .. } => operand_fact(input, state),
        MirCollectiveOp::Cat { input, .. } => {
            let mut fact = operand_fact(input, state);
            fact.shape = runmat_types::ShapeFact::Unknown;
            fact
        }
        MirCollectiveOp::FunctionalReduce { .. } => ValueFact::unknown(DynamicReason::RuntimeValue),
        MirCollectiveOp::Receive { .. } => ValueFact::unknown(DynamicReason::RuntimeValue),
        MirCollectiveOp::Probe { .. } => ValueFact::scalar(ValueKindFact::Logical),
    }
}

fn dynamic_value() -> ValueFact {
    ValueFact::unknown(DynamicReason::Unspecified)
}
