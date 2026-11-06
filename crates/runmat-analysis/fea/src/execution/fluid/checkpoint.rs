//! Original immutable temporal history. Decode yields claims, never restart authority.
mod admitted;
mod continuation;
mod profile;
use super::{
    invalid, AnalysisExecutionResult, FluidTransientBlockResult, FluidTransientWorkload,
    PreparedFluidTransient, LIMITS,
};
use crate::{
    admission::AnalysisRunIdentity,
    execution::AnalysisWorkloadIdentity,
    physics::cfd::canonical::{CanonicalFluidFields, CanonicalFluidInitialState},
    properties::CompiledPropertyMapIdentity,
};
pub use continuation::{FluidTransientBlockWorkload, PreparedFluidTransientContinuation};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};
const PREFIX: &[u8] = b"runmat-fluid-transient-checkpoint/v1\0";
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Scope {
    workload: AnalysisWorkloadIdentity,
    topology: Digest,
    formulation: AnalysisRunIdentity,
    properties: CompiledPropertyMapIdentity,
    property_audit: Digest,
    initial: CanonicalFluidInitialState,
}
impl Scope {
    fn derive(p: &PreparedFluidTransient<'_>) -> AnalysisExecutionResult<Self> {
        let initial = runmat_canonical_codec::profile_serialized(&p.initial, LIMITS)?;
        let required = initial
            .materialization_resident_bytes_estimate
            .checked_mul(2)
            .ok_or_else(|| invalid("fluid checkpoint scope initial heap overflow"))
            .and_then(|later| p.source.resident_with(later))?;
        super::source::require_bytes(required, p.workload.resident_ceiling()?)?;
        Ok(Self {
            workload: p.workload.identity()?,
            topology: Digest::from_bytes(*p.source.source.mesh_digest.bytes()),
            formulation: p.formulation.clone(),
            properties: p.source.property_identity.clone(),
            property_audit: p.source.property_audit_digest,
            initial: p.initial.clone(),
        })
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry<R> {
    start_interval: usize,
    end_interval: usize,
    charged_operator_work: u64,
    original: R,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Data<S, B> {
    schema_version: u16,
    scope: S,
    next_interval: usize,
    next_output_ordinal: usize,
    charged_operator_work: u64,
    blocks: B,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FluidTransientCheckpoint {
    data: Data<Scope, Vec<Entry<FluidTransientBlockResult>>>,
}
impl FluidTransientCheckpoint {
    pub fn next_interval(&self) -> usize {
        self.data.next_interval
    }
    /// Every original interval owns one output ordinal; observations are separate authority.
    pub fn next_output_ordinal(&self) -> usize {
        self.data.next_output_ordinal
    }
    /// Conservative numerical charges for retained successful blocks, not measured usage/retries.
    pub fn charged_operator_work(&self) -> u64 {
        self.data.charged_operator_work
    }
    /// Retained original intervals of every block, with schedule interval ordinals.
    pub fn retained_results(
        &self,
    ) -> impl Iterator<Item = &(usize, crate::physics::cfd::canonical::CanonicalFluidResult)> {
        self.data
            .blocks
            .iter()
            .flat_map(|b| b.original.retained_results())
    }
    /// Evidence of every accepted interval of every block.
    pub fn intervals(&self) -> impl Iterator<Item = &super::FluidIntervalEvidence> {
        self.data.blocks.iter().flat_map(|b| b.original.intervals())
    }
    fn terminal_fields(&self) -> CanonicalFluidFields<'_> {
        self.data
            .blocks
            .last()
            .and_then(|b| b.original.terminal())
            .map_or_else(
                || self.data.scope.initial.original_fields(),
                |r| r.original_fields(),
            )
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        Ok(runmat_canonical_codec::encode_contract(
            PREFIX,
            "fluid-transient-checkpoint",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(runmat_canonical_codec::decode_contract(
            PREFIX,
            "fluid-transient-checkpoint",
            bytes,
            LIMITS,
        )?)
    }
    pub fn identity(&self) -> AnalysisExecutionResult<Digest> {
        Ok(Digest::sha256(self.canonical_bytes()?))
    }
    pub fn recover(
        &self,
        workload: &FluidTransientWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        let p = workload.prepare(mesh)?;
        continuation::recover(&p, self).map(|_| ())
    }
}
impl FluidTransientWorkload {
    pub fn initial_checkpoint(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<FluidTransientCheckpoint> {
        let p = self.prepare(mesh)?;
        let scope = Scope::derive(&p)?;
        profile::require_complete_budget(&p, &scope)?;
        Ok(FluidTransientCheckpoint {
            data: Data {
                schema_version: 1,
                scope,
                next_interval: 0,
                next_output_ordinal: 0,
                charged_operator_work: 0,
                blocks: Vec::new(),
            },
        })
    }
}
