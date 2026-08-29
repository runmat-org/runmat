use runmat_types::{LabCount, LabRank, ParallelRegionId, SpmdLabRequirement};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    rename_all = "snake_case",
    tag = "kind",
    content = "value",
    deny_unknown_fields
)]
pub enum SpmdOutputValue {
    Value(crate::value::ValuePayload),
    Distributed(Box<crate::DistributedShardSnapshot>),
}

impl SpmdOutputValue {
    pub fn validate(&self) -> Result<(), crate::ContractError> {
        match self {
            Self::Value(value) => value.validate(crate::value::ValueLimits::default()),
            Self::Distributed(snapshot) => snapshot.validate(),
        }
    }
}

use crate::{ContractError, ExecutionScopeId, GangId, PoolHandle};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GangRequest {
    pub pool: PoolHandle,
    pub labs: SpmdLabRequirement,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GangHandle {
    pub id: GangId,
    pub scope_id: ExecutionScopeId,
    pub generation: u64,
    pub pool: PoolHandle,
    pub labs: LabCount,
}

impl GangHandle {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.generation == 0 || self.labs.0 == 0 {
            return Err(ContractError::invalid(
                "SPMD gang handle",
                "generation and lab count must be non-zero",
            ));
        }
        if self.scope_id != self.pool.scope_id {
            return Err(ContractError::invalid(
                "SPMD gang handle",
                "gang and pool must belong to the same execution scope",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GangSnapshot {
    pub handle: GangHandle,
    pub ranks: Vec<LabRank>,
}

impl GangSnapshot {
    pub fn validate(&self) -> Result<(), ContractError> {
        self.handle.validate()?;
        let expected = (1..=self.handle.labs.0).map(LabRank).collect::<Vec<_>>();
        if self.ranks != expected {
            return Err(ContractError::invalid(
                "SPMD gang snapshot",
                "ranks must be complete, one-based, unique, and canonically ordered",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpmdTaskContext {
    pub gang: GangHandle,
    pub region: ParallelRegionId,
    pub rank: LabRank,
}

impl SpmdTaskContext {
    pub fn validate(&self) -> Result<(), ContractError> {
        self.gang.validate()?;
        if self.rank.0 == 0 || self.rank.0 > self.gang.labs.0 {
            return Err(ContractError::invalid(
                "SPMD task context",
                "rank must be within the gang's one-based lab range",
            ));
        }
        Ok(())
    }
}
