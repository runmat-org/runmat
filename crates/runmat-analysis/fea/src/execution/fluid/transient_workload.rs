//! Portable fresh transient intent, original initial fields and source-bound cold numerical claims.
mod prepared;
mod result;
mod validation;
use super::{invalid, AnalysisExecutionResult, LIMITS};
use crate::{
    execution::{AnalysisWorkloadIdentity, InitialFieldBinding},
    physics::cfd::canonical::CanonicalFluidPolicy,
    topology::FiniteVolumeProjectionPolicy,
};
pub use prepared::PreparedFluidTransient;
pub use result::FluidTransientSolverResult;
use runmat_analysis_core::{AnalysisModel, ProcedureStepId};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};
const PREFIX: &[u8] = b"runmat-fluid-transient-workload/v1\0";
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidTransientWorkload {
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub policy: CanonicalFluidPolicy,
    pub geometry_policy: FiniteVolumeProjectionPolicy,
    pub initial_fields: Vec<InitialFieldBinding>,
}
impl FluidTransientWorkload {
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(runmat_canonical_codec::encode_contract(
            PREFIX,
            "fluid-transient-workload",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let result: Self = runmat_canonical_codec::decode_contract(
            PREFIX,
            "fluid-transient-workload",
            bytes,
            LIMITS,
        )?;
        result.validate()?;
        Ok(result)
    }
    pub fn identity(&self) -> AnalysisExecutionResult<AnalysisWorkloadIdentity> {
        AnalysisWorkloadIdentity::derive(
            "fluid-transient-workload",
            &Digest::sha256(self.canonical_bytes()?),
        )
    }
    pub fn prepare(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<PreparedFluidTransient<'_>> {
        let profile = runmat_canonical_codec::profile_serialized(self, LIMITS)?;
        super::source::require_bytes(
            profile.materialization_resident_bytes_estimate,
            self.resident_ceiling()?,
        )?;
        self.validate()?;
        let formulation =
            crate::admission::admit_canonical_fluid_transient(mesh).map_err(invalid)?;
        let source = super::source::PreparedFluidSource::prepare(
            self,
            &self.model,
            mesh,
            self.geometry_policy,
            self.policy,
            self.resident_ceiling()?,
        )?;
        let step = self.selected_step()?;
        let schedule = super::FluidTransientSchedule::derive(&self.model, step)?;
        let start = schedule.points_s()[0];
        let mut policy = self.policy;
        policy.set_resident_byte_budget(
            self.resident_ceiling()?
                .checked_sub(source.preparation_resident_bytes)
                .ok_or_else(|| invalid("transient source exhausts initial field ceiling"))?,
        );
        let initial_problem = crate::physics::cfd::canonical::CanonicalFluidProblem::prepare_at(
            source.input(&self.model, policy),
            crate::excitation::ProjectionSample::TimeSeconds {
                value: start,
                approach: runmat_analysis_core::SignalSampleApproach::At,
            },
        )
        .map_err(invalid)?;
        let initial = initial_problem
            .realize_initial_state(&self.model, step, &self.initial_fields)
            .map_err(invalid)?;
        Ok(PreparedFluidTransient {
            workload: self,
            source,
            initial,
            formulation,
        })
    }
    /// Local numerical execution; the generic native task must separately admit placement/resources.
    pub fn execute(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<FluidTransientSolverResult> {
        self.prepare(mesh)?.execute()
    }
}
