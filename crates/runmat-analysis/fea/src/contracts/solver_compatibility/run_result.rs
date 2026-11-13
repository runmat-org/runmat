use runmat_analysis_core::{AnalysisField, TemporalImplementation};
use serde::Deserialize;

use crate::admission::AnalysisRunIdentity;

use super::{validate_solver_formulation_compatibility, SolverFormulationCompatibilityError};
use crate::contracts::{
    ComputeBackend, ExecutionBackend, FeaRunResult, SolverMethod, SolverPreconditioner,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::contracts) struct FeaRunResultWire {
    identity: AnalysisRunIdentity,
    requested_backend: ComputeBackend,
    execution_backend: ExecutionBackend,
    solver_backend: crate::solve::backend::kind::LinearAlgebraBackendKind,
    solver_device_apply_k_ratio: f64,
    solver_method: SolverMethod,
    preconditioner: SolverPreconditioner,
    solver_host_sync_count: u32,
    accelerator_evidence: Vec<crate::solve::accelerator::AcceleratorSolveEvidence>,
    diagnostics: Vec<crate::diagnostics::FeaDiagnostic>,
    #[serde(default)]
    spatial_projection_evidence: Vec<crate::excitation::SpatialProjectionEvidence>,
    #[serde(default)]
    structural_static_convergence: Option<crate::contracts::StructuralStaticConvergence>,
    #[serde(default)]
    structural_static_balance: Option<Box<crate::solve::StructuralStaticBalanceEvidence>>,
    #[serde(default)]
    structural_constraint_result:
        Option<Box<crate::constraints::CanonicalStructuralConstraintResult>>,
    #[serde(default)]
    structural_connector_result: Option<Box<crate::connectors::CanonicalStructuralConnectorResult>>,
    #[serde(default)]
    observations: Vec<crate::results::observation::ObservationResult>,
    observation_histories: Vec<crate::results::observation::ObservationHistory>,
    series_reductions: Vec<crate::results::observation::SeriesReductionResult>,
    fields: Vec<AnalysisField>,
}

impl TryFrom<FeaRunResultWire> for FeaRunResult {
    type Error = SolverFormulationCompatibilityError;

    fn try_from(wire: FeaRunResultWire) -> Result<Self, Self::Error> {
        let result = Self {
            identity: wire.identity,
            requested_backend: wire.requested_backend,
            execution_backend: wire.execution_backend,
            solver_backend: wire.solver_backend,
            solver_device_apply_k_ratio: wire.solver_device_apply_k_ratio,
            solver_method: wire.solver_method,
            preconditioner: wire.preconditioner,
            solver_host_sync_count: wire.solver_host_sync_count,
            accelerator_evidence: wire.accelerator_evidence,
            diagnostics: wire.diagnostics,
            spatial_projection_evidence: wire.spatial_projection_evidence,
            structural_static_convergence: wire.structural_static_convergence,
            structural_static_balance: wire.structural_static_balance,
            structural_constraint_result: wire.structural_constraint_result,
            structural_connector_result: wire.structural_connector_result,
            observations: wire.observations,
            observation_histories: wire.observation_histories,
            series_reductions: wire.series_reductions,
            fields: wire.fields,
        };
        result.validate_solver_formulation()?;
        Ok(result)
    }
}

impl FeaRunResult {
    pub fn validate_solver_formulation(&self) -> Result<(), SolverFormulationCompatibilityError> {
        validate_solver_formulation_compatibility(
            &self.identity,
            self.solver_method,
            self.preconditioner,
        )?;
        let is_structural_static = self
            .identity
            .realization()
            .temporals
            .contains(&TemporalImplementation::StructuralStaticLinearSolve);
        match (is_structural_static, &self.structural_static_convergence) {
            (true, Some(evidence)) => evidence.validate()?,
            (true, None) => {
                return Err(SolverFormulationCompatibilityError::MissingStructuralStaticConvergence)
            }
            (false, Some(_)) => {
                return Err(
                    SolverFormulationCompatibilityError::UnexpectedStructuralStaticConvergence,
                )
            }
            (false, None) => {}
        }
        let has_structural_connector_state = is_structural_static
            || self
                .identity
                .realization()
                .temporals
                .contains(&TemporalImplementation::StructuralNewmarkAverageAcceleration);
        match (is_structural_static, &self.structural_static_balance) {
            (true, Some(evidence)) => evidence
                .validate()
                .map_err(|_| SolverFormulationCompatibilityError::InvalidStructuralStaticBalance)?,
            (true, None) => {
                return Err(SolverFormulationCompatibilityError::MissingStructuralStaticBalance)
            }
            (false, Some(_)) => {
                return Err(SolverFormulationCompatibilityError::UnexpectedStructuralStaticBalance)
            }
            (false, None) => {}
        }
        let has_structural_constraint_state = is_structural_static
            || self
                .identity
                .realization()
                .temporals
                .contains(&TemporalImplementation::StructuralQuasistaticNewtonEquilibrium);
        match (
            has_structural_constraint_state,
            &self.structural_constraint_result,
        ) {
            (true, Some(result)) => result.validate().map_err(|_| {
                SolverFormulationCompatibilityError::InvalidStructuralConstraintResult
            })?,
            (true, None) => {
                return Err(SolverFormulationCompatibilityError::MissingStructuralConstraintResult)
            }
            (false, Some(_)) => {
                return Err(
                    SolverFormulationCompatibilityError::UnexpectedStructuralConstraintResult,
                )
            }
            (false, None) => {}
        }
        match (
            has_structural_connector_state,
            &self.structural_connector_result,
        ) {
            (true, Some(result)) => result.validate().map_err(|_| {
                SolverFormulationCompatibilityError::InvalidStructuralConnectorResult
            })?,
            (true, None) => {
                return Err(SolverFormulationCompatibilityError::MissingStructuralConnectorResult)
            }
            (false, Some(_)) => {
                return Err(
                    SolverFormulationCompatibilityError::UnexpectedStructuralConnectorResult,
                )
            }
            (false, None) => {}
        }
        Ok(())
    }
}
