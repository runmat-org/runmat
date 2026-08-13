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
        MirDistributedOp::LocalPart { value } | MirDistributedOp::Materialize { value } => state
            .distributed
            .get(value)
            .cloned()
            .unwrap_or_else(dynamic_value),
        MirDistributedOp::Redistribute { value, scheme } => {
            let underlying = state
                .distributed
                .get(value)
                .cloned()
                .unwrap_or_else(dynamic_value);
            let Some(owner) = state.locals.iter().find_map(|local| match &local.fact {
                Some(ValueFact {
                    kind: ValueKindFact::Distributed(distributed),
                    ..
                }) if distributed.id == *value => Some(distributed.owner),
                _ => None,
            }) else {
                return dynamic_value();
            };
            ValueFact::scalar(ValueKindFact::Distributed(runmat_types::DistributedFact {
                id: *value,
                owner,
                scheme: Some(scheme.clone()),
                value: Box::new(underlying),
                materializable: true,
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
        MirCollectiveOp::Broadcast { input, .. }
        | MirCollectiveOp::Gather { input, .. }
        | MirCollectiveOp::Scatter { input, .. }
        | MirCollectiveOp::AllGather { input, .. }
        | MirCollectiveOp::Reduce { input, .. }
        | MirCollectiveOp::AllReduce { input, .. } => operand_fact(input, state),
        MirCollectiveOp::Receive { .. } => ValueFact::unknown(DynamicReason::RuntimeValue),
    }
}

fn dynamic_value() -> ValueFact {
    ValueFact::unknown(DynamicReason::Unspecified)
}
