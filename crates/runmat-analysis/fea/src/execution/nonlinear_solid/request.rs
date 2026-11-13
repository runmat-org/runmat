use super::*;
use runmat_canonical_codec::decode_contract;
use serde::{Deserialize, Serialize};

const PREFIX: &[u8] = b"runmat-nonlinear-solid-block-request/v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonlinearSolidBlockReference {
    pub artifact_digest: Digest,
    pub controller_digest: Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonlinearSolidBlockRequest {
    pub workload: AnalysisWorkloadIdentity,
    pub topology: Digest,
    pub block_ordinal: u64,
    /// Ordinal of the first new accepted output; zero belongs to the initial state.
    pub first_output_ordinal: u64,
    pub accepted_block_limit: usize,
    pub predecessor: Option<NonlinearSolidBlockReference>,
}

impl NonlinearSolidBlockRequest {
    /// Derive the first original output interval; callers do not choose lineage ordinals.
    pub fn initial(
        workload: &NonlinearSolidWorkload,
        topology: Digest,
        accepted_block_limit: usize,
    ) -> AnalysisExecutionResult<Self> {
        let request = Self {
            workload: workload.identity()?,
            topology,
            block_ordinal: 0,
            first_output_ordinal: 1,
            accepted_block_limit,
            predecessor: None,
        };
        request.validate_for(workload, topology, None)?;
        Ok(request)
    }

    pub fn continue_original(
        workload: &NonlinearSolidWorkload,
        previous: &NonlinearSolidBlockArtifact,
        accepted_block_limit: usize,
    ) -> AnalysisExecutionResult<Self> {
        let request = Self {
            workload: workload.identity()?,
            topology: previous.request.topology,
            block_ordinal: previous
                .request
                .block_ordinal
                .checked_add(1)
                .ok_or_else(|| invalid("nonlinear block ordinal overflow"))?,
            first_output_ordinal: previous
                .last_output_ordinal()?
                .checked_add(1)
                .ok_or_else(|| invalid("nonlinear output ordinal overflow"))?,
            accepted_block_limit,
            predecessor: Some(NonlinearSolidBlockReference {
                artifact_digest: previous.identity,
                controller_digest: previous.checkpoint.content_digest(),
            }),
        };
        request.validate_for(workload, request.topology, Some(previous))?;
        Ok(request)
    }
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.accepted_block_limit == 0
            || self.first_output_ordinal == 0
            || (self.block_ordinal == 0) != self.predecessor.is_none()
            || (self.block_ordinal == 0 && self.first_output_ordinal != 1)
        {
            return Err(invalid(
                "nonlinear block has invalid initial/continuation ordinals",
            ));
        }
        Ok(())
    }
    pub fn identity(&self) -> AnalysisExecutionResult<Digest> {
        self.validate()?;
        digest("nonlinear-solid-block-request", self)
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "nonlinear-solid-block-request",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let request: Self =
            decode_contract(PREFIX, "nonlinear-solid-block-request", bytes, LIMITS)?;
        request.validate()?;
        Ok(request)
    }
    pub fn validate_for(
        &self,
        workload: &NonlinearSolidWorkload,
        topology: Digest,
        predecessor: Option<&NonlinearSolidBlockArtifact>,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        if self.workload != workload.identity()?
            || self.topology != topology
            || self.accepted_block_limit > workload.controls.max_accepted_increments
        {
            return Err(invalid(
                "nonlinear block differs from original workload/mesh/budget",
            ));
        }
        match (&self.predecessor, predecessor) {
            (None, None) => Ok(()),
            (Some(expected), Some(previous)) => {
                previous.validate()?;
                if previous.complete
                    || previous.request.workload != self.workload
                    || previous.request.topology != self.topology
                    || previous.identity != expected.artifact_digest
                    || previous.checkpoint.content_digest() != expected.controller_digest
                    || previous.request.block_ordinal.checked_add(1) != Some(self.block_ordinal)
                    || previous.last_output_ordinal()?.checked_add(1)
                        != Some(self.first_output_ordinal)
                {
                    return Err(invalid(
                        "nonlinear block does not continue its exact original predecessor",
                    ));
                }
                Ok(())
            }
            _ => Err(invalid(
                "nonlinear block predecessor is missing or unexpected",
            )),
        }
    }
}
