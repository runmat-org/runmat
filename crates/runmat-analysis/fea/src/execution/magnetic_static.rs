//! Portable physical magnetostatics; neither resource admission nor native dispatch.
use super::{
    AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkloadIdentity, SequenceIdentity,
    StepIdentity,
};
use crate::admission::AnalysisRunIdentity;
use runmat_analysis_core::{AnalysisModel, ProcedureStepId};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

mod admitted;
pub(in crate::execution) mod fields;
mod receipt;
pub use receipt::{
    MagneticNumericalReceipt, MagneticNumericalReceiptArtifact, MAGNETIC_RECEIPT_MAX_ENCODED_BYTES,
};
mod publication;
mod result;
pub use fields::magnetic_native_field_definition_inventory;
pub use fields::magnetic_native_field_definitions;
pub use result::MagnetostaticSolverResult;
#[cfg(test)]
mod tests;
const PREFIX: &[u8] = b"runmat-magnetostatic-workload/v1\0";
const LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MagnetostaticWorkload {
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub relative_residual_target: f64,
}

impl MagnetostaticWorkload {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        crate::physics::magnetic::validate_canonical_input(
            &self.model,
            self.step_id.as_str(),
            self.relative_residual_target,
        )
        .map_err(invalid)?;
        Ok(())
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "magnetostatic-workload",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let value: Self = decode_contract(PREFIX, "magnetostatic-workload", bytes, LIMITS)?;
        value.validate()?;
        Ok(value)
    }
    pub fn identity(&self) -> AnalysisExecutionResult<AnalysisWorkloadIdentity> {
        AnalysisWorkloadIdentity::derive(
            "magnetostatic-workload",
            &Digest::sha256(self.canonical_bytes()?),
        )
    }
    pub fn producer_identity(&self) -> AnalysisExecutionResult<StepIdentity> {
        self.validate()?;
        StepIdentity::from_solver_contract(
            SequenceIdentity::from_solver_contract(&self.model.sequence)?,
            &self.model.sequence.steps[0],
        )
    }
    pub fn formulation(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<AnalysisRunIdentity> {
        self.formulation_with_sampling(mesh, None)
    }
    /// As [`Self::formulation`], resolving native selections on `sampling` when the caller
    /// already holds the mesh's validated layout (no second projection is derived).
    pub(crate) fn formulation_with_sampling(
        &self,
        mesh: &SolverMeshArtifact,
        sampling: Option<&crate::results::field::SampledVolumeFieldTopology>,
    ) -> AnalysisExecutionResult<AnalysisRunIdentity> {
        self.validate()?;
        // Resolve authored native selections before any numerical assembly/solve.
        // Admission must not defer an invalid quantity or selector until publication.
        if !self.model.sequence.steps[0].observations.is_empty() {
            let derived;
            let sampling = match sampling {
                Some(sampling) => sampling,
                None => {
                    derived = crate::results::field::SampledVolumeFieldTopology::derive(
                        mesh,
                        crate::results::field::VolumeFieldSamplingRule::TetrahedronKeastFifteen,
                    )
                    .map_err(invalid)?;
                    &derived
                }
            };
            crate::results::observation::resolve_sampled_observation_plan(
                &self.model.sequence.steps[0],
                sampling,
                mesh,
                &magnetic_native_field_definition_inventory()?,
                crate::results::observation::ObservationLimits::for_step(
                    &self.model.sequence.steps[0],
                ),
            )
            .map_err(invalid)?;
        }
        crate::admission::admit_canonical_magnetostatic(mesh).map_err(invalid)
    }
    /// Serial numerical execution only. No scheduler reservation or state publication implied.
    pub fn execute(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<MagnetostaticSolverResult> {
        let input = crate::physics::magnetic::compile_canonical_input(
            &self.model,
            mesh,
            self.step_id.as_str(),
            self.relative_residual_target,
        )
        .map_err(invalid)?;
        let sampling = crate::results::field::SampledVolumeFieldTopology::derive_from_projection(
            mesh,
            &input.projection,
            crate::results::field::VolumeFieldSamplingRule::TetrahedronKeastFifteen,
        )
        .map_err(invalid)?;
        let formulation = self.formulation_with_sampling(mesh, Some(&sampling))?;
        let response = crate::physics::magnetic::solve_grounded_volume(
            &input.projection,
            &input.grounded_faces,
            &input.prescribed_potential,
            &input.volumes,
            self.relative_residual_target,
        )
        .map_err(magnetic_failure)?;
        drop(input);
        MagnetostaticSolverResult::new(self, mesh, formulation, sampling, response)
    }
}

fn magnetic_failure(error: crate::physics::magnetic::MagnetostaticError) -> AnalysisExecutionError {
    match error {
        crate::physics::magnetic::MagnetostaticError::Accounting(cause) => {
            AnalysisExecutionError::Usage(cause)
        }
        other => invalid(other),
    }
}

fn invalid(error: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(error.to_string())
}
