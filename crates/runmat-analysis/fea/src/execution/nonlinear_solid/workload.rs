use super::*;
use crate::{
    pipeline::nonlinear::{run_nonlinear_block, NonlinearBlockInput},
    solve::nonlinear::NonlinearSolveOptions,
    ComputeBackend,
};
use runmat_analysis_core::AnalysisModel;
use runmat_canonical_codec::decode_contract;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

const PREFIX: &[u8] = b"runmat-nonlinear-solid-workload/v1\0";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonlinearSolidWorkload {
    pub model: AnalysisModel,
    /// Canonical mesh is a separate immutable numerical input.
    pub controls: NonlinearSolveOptions,
}

impl NonlinearSolidWorkload {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        runmat_analysis_core::validate_model(&self.model).map_err(invalid)?;
        crate::solve::nonlinear::require_fixed_structural_nonlinear(&self.model)
            .map_err(invalid)?;
        self.controls.validate_controls().map_err(invalid)?;
        if self.controls.solver_mesh.is_some()
            || (self.controls.contact_enforcement.is_some() && self.model.interfaces.is_empty())
        {
            return Err(invalid(
                "portable J2 requires a separate mesh; contact enforcement needs an authored contact interface",
            ));
        }
        Ok(())
    }

    pub fn identity(&self) -> AnalysisExecutionResult<AnalysisWorkloadIdentity> {
        self.validate()?;
        AnalysisWorkloadIdentity::derive("nonlinear-solid-workload", self)
    }

    pub fn producer_identity(&self) -> AnalysisExecutionResult<super::super::StepIdentity> {
        self.validate()?;
        let sequence = super::super::SequenceIdentity::from_solver_contract(&self.model.sequence)?;
        super::super::StepIdentity::from_solver_contract(sequence, &self.model.sequence.steps[0])
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "nonlinear-solid-workload",
            self,
            LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let workload: Self = decode_contract(PREFIX, "nonlinear-solid-workload", bytes, LIMITS)?;
        workload.validate()?;
        Ok(workload)
    }

    /// Mesh ownership moves into numerical options; no second full mesh is retained.
    pub fn execute_block(
        &self,
        mesh: SolverMeshArtifact,
        request: &NonlinearSolidBlockRequest,
        predecessor: Option<&NonlinearSolidBlockArtifact>,
    ) -> AnalysisExecutionResult<NonlinearSolidBlockArtifact> {
        self.validate()?;
        request.validate_for(
            self,
            Digest::from_bytes(*mesh.canonical_digest.bytes()),
            predecessor,
        )?;
        let input = match predecessor {
            Some(previous) => NonlinearBlockInput::Continue {
                checkpoint: &previous.checkpoint,
                expected_digest: request
                    .predecessor
                    .as_ref()
                    .ok_or_else(|| invalid("J2 predecessor reference missing"))?
                    .controller_digest,
            },
            None => NonlinearBlockInput::Fresh,
        };
        let mut options = self.controls.clone();
        options.solver_mesh = Some(mesh);
        let block = run_nonlinear_block(
            &self.model,
            ComputeBackend::Cpu,
            &options,
            input,
            request.accepted_block_limit,
        )
        .map_err(invalid)?;
        NonlinearSolidBlockArtifact::new(request.clone(), block)
    }
}
