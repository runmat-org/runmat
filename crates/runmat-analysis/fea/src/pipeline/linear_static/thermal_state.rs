//! Static physical consumption of an immutable, topology-bound thermal solution.

use runmat_analysis_core::AnalysisModel;
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;

#[cfg(test)]
mod qualification;
#[cfg(test)]
mod tests;

use crate::{
    contracts::{ComputeBackend, FeaRunError, FeaRunResult, LinearStaticSolveOptions},
    execution::{CanonicalSemanticState, CanonicalStateTransition, SolverResultIdentity},
};

pub struct ThermoelasticStaticResult {
    pub run: FeaRunResult,
    pub thermal_transition: CanonicalStateTransition,
    pub solver_result_identity: SolverResultIdentity,
    /// Physical initial-stress operator from this exact solve and thermal transition.
    pub linearization: crate::pipeline::structural_linearization::StructuralLinearizationArtifact,
}

/// Consume the exact retained nodal temperature field, not a prescribed profile or solver guess.
pub fn run_linear_static_from_thermal_state(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: LinearStaticSolveOptions,
    source: &CanonicalSemanticState,
) -> Result<ThermoelasticStaticResult, FeaRunError> {
    let result = super::run_linear_static_with_constraint_plan(
        model,
        backend,
        options,
        None,
        None,
        Some(source),
    )?;
    let thermal_transition = result.thermal_transition.ok_or_else(|| {
        FeaRunError::Assembly(
            "solved thermoelastic execution did not retain its source transition".into(),
        )
    })?;
    let content = encode_contract(
        b"runmat-analysis-thermoelastic-result/v1\0",
        "solved-thermal-static-result",
        &(
            model.solver_identity_contract(),
            &thermal_transition,
            &result.run.identity,
            &result.run.fields,
        ),
        CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64),
    )
    .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let structural_result_content = Digest::sha256(content);
    let solver_result_identity = SolverResultIdentity::from_step_result(
        thermal_transition.consumer,
        structural_result_content,
    )
    .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let linearization = crate::pipeline::structural_linearization::StructuralLinearizationArtifact::from_thermal_solution(
        &result.summary,
        result.run.identity.clone(),
        &result.solution,
        thermal_transition.clone(),
        structural_result_content,
    )
    .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    Ok(ThermoelasticStaticResult {
        run: result.run,
        thermal_transition,
        solver_result_identity,
        linearization,
    })
}
