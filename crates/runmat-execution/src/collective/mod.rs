use runmat_types::{CollectiveId, LabRank, OperatorKind};
use serde::{Deserialize, Serialize};

use crate::value::ValuePayload;
use crate::{ContractError, SpmdTaskContext};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct CollectiveSequence(pub u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct CollectiveMessageTag(pub u64);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub struct ReceiveSelection {
    pub source: Option<LabRank>,
    pub tag: Option<CollectiveMessageTag>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributedBuildContribution {
    pub value: runmat_types::ValueFact,
    pub local_shape: Vec<u64>,
    pub codistributor: Option<ValuePayload>,
}

impl DistributedBuildContribution {
    fn validate(&self) -> Result<(), ContractError> {
        if self.local_shape.is_empty() {
            return Err(ContractError::invalid(
                "distributed build contribution",
                "local shape must have at least one dimension",
            ));
        }
        if let Some(codistributor) = &self.codistributor {
            codistributor.validate(crate::value::ValueLimits::default())?;
            if matches!(
                codistributor,
                ValuePayload::Distributed(_) | ValuePayload::Composite(_)
            ) {
                return Err(ContractError::invalid(
                    "distributed build contribution",
                    "codistributor metadata cannot contain a live execution handle",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "operation", deny_unknown_fields)]
pub enum CollectiveInvocation {
    Barrier,
    Broadcast {
        root: LabRank,
        value: Option<ValuePayload>,
    },
    Gather {
        root: LabRank,
        value: ValuePayload,
    },
    Scatter {
        root: LabRank,
        values: Option<Vec<ValuePayload>>,
    },
    AllGather {
        value: ValuePayload,
    },
    /// Internal compiler-owned agreement check for replicated construction.
    /// Only the canonical logical digest crosses the coordination boundary;
    /// each worker retains its local value payload.
    AssertEqual {
        digest: crate::Digest,
    },
    /// Internal compiler-owned coordination for `codistributed.build`.
    /// Only facts, shapes, and immutable codistributor metadata cross this
    /// boundary; each partition payload remains on its owning worker.
    DistributedBuild {
        contribution: Box<DistributedBuildContribution>,
        validate_across_workers: bool,
    },
    Reduce {
        root: Option<LabRank>,
        operator: OperatorKind,
        value: ValuePayload,
    },
    Cat {
        root: Option<LabRank>,
        dimension: u32,
        value: ValuePayload,
    },
    FunctionalReduce {
        root: Option<LabRank>,
        reducer: ValuePayload,
        value: ValuePayload,
    },
    Send {
        destination: LabRank,
        tag: CollectiveMessageTag,
        value: ValuePayload,
    },
    SendReceive {
        destination: Option<LabRank>,
        source: Option<LabRank>,
        tag: CollectiveMessageTag,
        value: ValuePayload,
    },
    Receive {
        selection: ReceiveSelection,
    },
    Probe {
        selection: ReceiveSelection,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollectiveRequest {
    pub context: SpmdTaskContext,
    pub id: CollectiveId,
    pub sequence: CollectiveSequence,
    pub invocation: CollectiveInvocation,
}

impl CollectiveRequest {
    pub fn validate(&self) -> Result<(), ContractError> {
        self.context.validate()?;
        if self.id.region != self.context.region {
            return Err(ContractError::invalid(
                "collective request",
                "collective identity does not belong to the executing SPMD region",
            ));
        }
        validate_invocation(&self.context, &self.invocation)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "outcome", deny_unknown_fields)]
pub enum CollectiveResponse {
    Complete,
    Value {
        value: ValuePayload,
    },
    Values {
        values: Vec<ValuePayload>,
    },
    Agreement {
        equal: bool,
    },
    DistributedBuild {
        contributions: Vec<DistributedBuildContribution>,
    },
    /// Ordered, rank-stable inputs for a runtime-owned reduction. The
    /// execution coordinator never interprets language values or duplicates
    /// builtin arithmetic semantics.
    ReductionInputs {
        operator: OperatorKind,
        values: Vec<ValuePayload>,
    },
    ConcatenationInputs {
        dimension: u32,
        values: Vec<ValuePayload>,
    },
    FunctionalReductionInputs {
        reducer: ValuePayload,
        values: Vec<ValuePayload>,
    },
    Received {
        value: ValuePayload,
        source: LabRank,
        tag: CollectiveMessageTag,
    },
    Probe {
        available: bool,
    },
}

fn validate_invocation(
    context: &SpmdTaskContext,
    invocation: &CollectiveInvocation,
) -> Result<(), ContractError> {
    let validate_rank = |rank: LabRank| {
        if rank.0 == 0 || rank.0 > context.gang.labs.0 {
            Err(ContractError::invalid(
                "collective rank",
                "rank must be within the gang's one-based lab range",
            ))
        } else {
            Ok(())
        }
    };
    match invocation {
        CollectiveInvocation::Broadcast { root, .. }
        | CollectiveInvocation::Gather { root, .. }
        | CollectiveInvocation::Scatter { root, .. } => validate_rank(*root),
        CollectiveInvocation::Reduce { root, .. } => {
            root.map(validate_rank).transpose().map(|_| ())
        }
        CollectiveInvocation::Cat {
            root, dimension, ..
        } => {
            if *dimension == 0 {
                return Err(ContractError::invalid(
                    "collective concatenation",
                    "dimension must be one-based",
                ));
            }
            root.map(validate_rank).transpose().map(|_| ())
        }
        CollectiveInvocation::FunctionalReduce { root, reducer, .. } => {
            reducer.validate(crate::value::ValueLimits::default())?;
            root.map(validate_rank).transpose().map(|_| ())
        }
        CollectiveInvocation::Send { destination, .. } => validate_rank(*destination),
        CollectiveInvocation::SendReceive {
            destination,
            source,
            ..
        } => {
            destination.map(validate_rank).transpose()?;
            source.map(validate_rank).transpose()?;
            Ok(())
        }
        CollectiveInvocation::Receive { selection } | CollectiveInvocation::Probe { selection } => {
            selection.source.map(validate_rank).transpose().map(|_| ())
        }
        CollectiveInvocation::DistributedBuild { contribution, .. } => contribution.validate(),
        CollectiveInvocation::Barrier
        | CollectiveInvocation::AllGather { .. }
        | CollectiveInvocation::AssertEqual { .. } => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExecutionScopeId, GangHandle, GangId, PoolHandle, PoolId};
    use runmat_types::{LabCount, ParallelRegionId, ProgramFunctionId, RegionId};

    fn context() -> SpmdTaskContext {
        let scope_id = ExecutionScopeId::derive(&[b"collective-test"]);
        SpmdTaskContext {
            gang: GangHandle {
                id: GangId::derive(&[b"gang"]),
                scope_id,
                generation: 1,
                pool: PoolHandle {
                    id: PoolId::derive(&[b"pool"]),
                    scope_id,
                    generation: 1,
                },
                labs: LabCount(3),
            },
            region: ParallelRegionId(RegionId {
                function: ProgramFunctionId(8),
                ordinal: 2,
            }),
            rank: LabRank(2),
        }
    }

    #[test]
    fn dynamic_collective_ranks_are_checked_against_the_admitted_gang() {
        let context = context();
        let mut request = CollectiveRequest {
            id: CollectiveId {
                region: context.region,
                ordinal: 4,
            },
            context,
            sequence: CollectiveSequence(7),
            invocation: CollectiveInvocation::Receive {
                selection: ReceiveSelection {
                    source: Some(LabRank(3)),
                    tag: Some(CollectiveMessageTag(11)),
                },
            },
        };
        request.validate().expect("rank belongs to gang");
        request.invocation = CollectiveInvocation::Receive {
            selection: ReceiveSelection {
                source: Some(LabRank(4)),
                tag: None,
            },
        };
        assert!(request.validate().is_err());
    }

    #[test]
    fn collective_identity_is_fenced_to_the_spmd_region() {
        let context = context();
        let request = CollectiveRequest {
            id: CollectiveId {
                region: ParallelRegionId(RegionId {
                    function: ProgramFunctionId(8),
                    ordinal: 9,
                }),
                ordinal: 1,
            },
            context,
            sequence: CollectiveSequence(1),
            invocation: CollectiveInvocation::Barrier,
        };
        assert!(request.validate().is_err());
    }
}
