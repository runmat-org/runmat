use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, StructuralTransientBlockRequestIdentity,
    StructuralTransientCheckpoint, StructuralTransientCheckpointIdentity,
    StructuralTransientCheckpointScope, StructuralTransientSolverCounters,
};

const TRANSIENT_BLOCK_PREFIX: &[u8] = b"runmat-structural-transient-block/v1\0";
const TRANSIENT_BLOCK_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "StructuralTransientBlockRequestWire")]
pub struct StructuralTransientBlockRequest {
    pub schema_version: u16,
    pub identity: StructuralTransientBlockRequestIdentity,
    pub scope: StructuralTransientCheckpointScope,
    pub predecessor: Option<StructuralTransientCheckpointIdentity>,
    pub block_ordinal: u64,
    pub start_schedule_ordinal: u64,
    pub terminal_schedule_ordinal: u64,
    pub start_counters: StructuralTransientSolverCounters,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralTransientBlockRequestWire {
    schema_version: u16,
    identity: StructuralTransientBlockRequestIdentity,
    scope: StructuralTransientCheckpointScope,
    predecessor: Option<StructuralTransientCheckpointIdentity>,
    block_ordinal: u64,
    start_schedule_ordinal: u64,
    terminal_schedule_ordinal: u64,
    start_counters: StructuralTransientSolverCounters,
}

impl TryFrom<StructuralTransientBlockRequestWire> for StructuralTransientBlockRequest {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralTransientBlockRequestWire) -> Result<Self, Self::Error> {
        let request = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            scope: wire.scope,
            predecessor: wire.predecessor,
            block_ordinal: wire.block_ordinal,
            start_schedule_ordinal: wire.start_schedule_ordinal,
            terminal_schedule_ordinal: wire.terminal_schedule_ordinal,
            start_counters: wire.start_counters,
        };
        request.validate()?;
        Ok(request)
    }
}

impl StructuralTransientBlockRequest {
    pub fn new(
        scope: StructuralTransientCheckpointScope,
        predecessor: Option<&StructuralTransientCheckpoint>,
        terminal_schedule_ordinal: u64,
    ) -> AnalysisExecutionResult<Self> {
        let predecessor_identity = predecessor.map(|value| value.identity);
        let block_ordinal = match predecessor {
            Some(value) => {
                value
                    .block_ordinal
                    .checked_add(1)
                    .ok_or(AnalysisExecutionError::Limit(
                        "structural transient block request ordinal overflow",
                    ))?
            }
            None => 0,
        };
        let start_schedule_ordinal =
            predecessor.map_or(0, |value| value.continuation.schedule_ordinal);
        let start_counters = predecessor
            .map(|value| value.counters.clone())
            .unwrap_or_default();
        let request = Self {
            schema_version: 1,
            identity: block_request_identity(BlockRequestIdentityInput {
                schema_version: 1,
                scope: &scope,
                predecessor: predecessor_identity,
                block_ordinal,
                start_schedule_ordinal,
                terminal_schedule_ordinal,
                start_counters: &start_counters,
            })?,
            scope,
            predecessor: predecessor_identity,
            block_ordinal,
            start_schedule_ordinal,
            terminal_schedule_ordinal,
            start_counters,
        };
        request.validate_against_predecessor(predecessor)?;
        Ok(request)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.scope.validate()?;
        if self.schema_version != 1
            || (self.block_ordinal == 0) != self.predecessor.is_none()
            || (self.predecessor.is_none()
                && (self.start_schedule_ordinal != 0
                    || self.start_counters != StructuralTransientSolverCounters::default()))
            || self.start_counters.device_apply_k_count
                > self.start_counters.device_apply_k_attempt_count
            || self.terminal_schedule_ordinal <= self.start_schedule_ordinal
            || usize::try_from(self.terminal_schedule_ordinal)
                .ok()
                .is_none_or(|ordinal| ordinal >= self.scope.schedule.points.len())
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural transient block has invalid lineage or schedule bounds".into(),
            ));
        }
        if self.identity
            != block_request_identity(BlockRequestIdentityInput {
                schema_version: self.schema_version,
                scope: &self.scope,
                predecessor: self.predecessor,
                block_ordinal: self.block_ordinal,
                start_schedule_ordinal: self.start_schedule_ordinal,
                terminal_schedule_ordinal: self.terminal_schedule_ordinal,
                start_counters: &self.start_counters,
            })?
        {
            return Err(AnalysisExecutionError::Identity(
                "structural transient block request identity does not converge",
            ));
        }
        Ok(())
    }

    pub fn validate_against_predecessor(
        &self,
        predecessor: Option<&StructuralTransientCheckpoint>,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        match (self.predecessor, predecessor) {
            (None, None) => Ok(()),
            (Some(expected), Some(predecessor)) => {
                predecessor.validate()?;
                let expected_block = predecessor.block_ordinal.checked_add(1).ok_or(
                    AnalysisExecutionError::Limit(
                        "structural transient block request ordinal overflow",
                    ),
                )?;
                if expected != predecessor.identity
                    || self.scope != predecessor.scope
                    || self.block_ordinal != expected_block
                    || self.start_schedule_ordinal != predecessor.continuation.schedule_ordinal
                    || self.start_counters != predecessor.counters
                {
                    return Err(AnalysisExecutionError::Identity(
                        "structural transient block does not continue its exact predecessor",
                    ));
                }
                Ok(())
            }
            _ => Err(AnalysisExecutionError::Identity(
                "structural transient block predecessor object differs from its request",
            )),
        }
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            TRANSIENT_BLOCK_PREFIX,
            "structural-transient-block-request",
            self,
            TRANSIENT_BLOCK_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            TRANSIENT_BLOCK_PREFIX,
            "structural-transient-block-request",
            bytes,
            TRANSIENT_BLOCK_LIMITS,
        )?)
    }
}

#[derive(Serialize)]
struct BlockRequestIdentityInput<'a> {
    schema_version: u16,
    scope: &'a StructuralTransientCheckpointScope,
    predecessor: Option<StructuralTransientCheckpointIdentity>,
    block_ordinal: u64,
    start_schedule_ordinal: u64,
    terminal_schedule_ordinal: u64,
    start_counters: &'a StructuralTransientSolverCounters,
}

fn block_request_identity(
    input: BlockRequestIdentityInput<'_>,
) -> AnalysisExecutionResult<StructuralTransientBlockRequestIdentity> {
    StructuralTransientBlockRequestIdentity::derive("structural-transient-block-request", &input)
}
