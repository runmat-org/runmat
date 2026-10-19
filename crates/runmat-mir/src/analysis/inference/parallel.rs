use runmat_types::{
    codistributor_fact, CodistributorClass, DynamicReason, NumericClass, NumericDomain,
    NumericFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

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
        MirDistributedOp::Codistributed {
            id, owner, input, ..
        } => {
            let value = operand_fact(input, state);
            state.distributed.insert(*id, value.clone());
            ValueFact::scalar(ValueKindFact::Distributed(runmat_types::DistributedFact {
                id: *id,
                owner: *owner,
                scheme: None,
                value: Box::new(value),
                materializable: true,
            }))
        }
        MirDistributedOp::Build {
            id,
            owner,
            local_part: input,
            ..
        } => {
            let mut value = operand_fact(input, state);
            value.shape = value
                .shape
                .rank()
                .map_or(ShapeFact::Unknown, |rank| ShapeFact::Ranked { rank });
            value.storage = StorageFact::Unknown;
            state.distributed.insert(*id, value.clone());
            ValueFact::scalar(ValueKindFact::Distributed(runmat_types::DistributedFact {
                id: *id,
                owner: *owner,
                scheme: None,
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
        MirDistributedOp::Codistributor { value } => {
            let class = match operand_fact(value, state).kind {
                ValueKindFact::Distributed(distributed) => distributed
                    .scheme
                    .as_ref()
                    .and_then(CodistributorClass::from_scheme),
                _ => None,
            };
            codistributor_fact(class)
        }
        MirDistributedOp::GlobalIndices {
            requested_outputs, ..
        } => {
            let kind = ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            });
            if *requested_outputs == 0 {
                ValueFact::scalar(ValueKindFact::Void)
            } else {
                ValueFact::proven(
                    kind,
                    ShapeFact::from(vec![Some(1), None]),
                    StorageFact::Dense,
                )
            }
        }
        MirDistributedOp::Redistribute { value, .. } => {
            let ValueKindFact::Distributed(distributed) = operand_fact(value, state).kind else {
                return dynamic_value();
            };
            ValueFact::scalar(ValueKindFact::Distributed(runmat_types::DistributedFact {
                id: distributed.id,
                owner: distributed.owner,
                scheme: None,
                value: distributed.value,
                materializable: distributed.materializable,
            }))
        }
    }
}

pub(crate) fn distributed_output_sequence(
    operation: &crate::parallel::MirDistributedOp,
) -> Option<runmat_types::ValueSequenceFact> {
    let crate::parallel::MirDistributedOp::GlobalIndices {
        requested_outputs, ..
    } = operation
    else {
        return None;
    };
    let index = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    Some(runmat_types::ValueSequenceFact::fixed(
        (0..*requested_outputs).map(|_| index.clone()).collect(),
    ))
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
