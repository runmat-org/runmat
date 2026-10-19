//! Terminal observations over original nonlinear recovered fields; no constitutive re-evaluation.
use super::*;
use crate::{
    constraints::CanonicalStructuralConstraintPlan,
    post::fields::take_structural_observable_fields,
    results::{
        field::{RecoveredFieldCatalog, StructuralRecoveredFieldTopology},
        observation::{
            evaluate_structural_observations, resolve_structural_observation_plan,
            ObservationLimits,
        },
    },
};

/// Shared admitted study envelope. Whole-history/event sources need their separate carrier.
pub fn require_nonlinear_terminal_field_outputs(
    step: &runmat_analysis_core::ProcedureStep,
) -> Result<(), FeaRunError> {
    if step.observations.iter().any(|request| {
        !matches!(request,
        ObservationRequest::Field(spec) if spec.sampling == SamplingPolicy::Final)
    }) {
        return Err(FeaRunError::Observation(
            "governed nonlinear blocks currently admit final fields; sampled/history observations require original whole-history evaluation".into(),
        ));
    }
    Ok(())
}

pub(super) fn evaluate(
    model: &AnalysisModel,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    step: &runmat_analysis_core::ProcedureStep,
    constraints: &CanonicalStructuralConstraintPlan,
    run: &mut FeaRunResult,
) -> Result<(), FeaRunError> {
    if step.observations.iter().any(|request| {
        matches!(request,
        ObservationRequest::Field(spec) if spec.sampling != SamplingPolicy::Final)
    }) {
        return Err(FeaRunError::Observation(
            "nonlinear sampled fields require original whole-history evaluation".into(),
        ));
    }
    if !step
        .observations
        .iter()
        .any(|request| matches!(request, ObservationRequest::Field(_)))
    {
        return Ok(());
    }
    // Reuse original source storage by move. Never recover elastic stress from displacement.
    let recovered = take_structural_observable_fields(&mut run.fields).map_err(error)?;
    let topology = StructuralRecoveredFieldTopology::derive(model, Some(mesh)).map_err(error)?;
    let catalog = RecoveredFieldCatalog::new_for_topology(
        topology.identity.discretization_digest(),
        topology.entity_counts,
        recovered,
    )
    .map_err(error)?;
    let definitions = catalog
        .fields()
        .iter()
        .map(|field| field.definition().clone())
        .collect::<Vec<_>>();
    let mut terminal = step.clone();
    terminal.observations.retain(|request| {
        matches!(request,
        ObservationRequest::Field(spec) if spec.sampling == SamplingPolicy::Final)
    });
    let plan = resolve_structural_observation_plan(
        &terminal,
        model,
        Some(mesh),
        constraints,
        &definitions,
        ObservationLimits::default(),
    )
    .map_err(error)?;
    run.observations =
        evaluate_structural_observations(&plan, model, Some(mesh), &catalog).map_err(error)?;
    run.fields.extend(
        catalog
            .into_fields()
            .into_iter()
            .map(|field| field.into_storage()),
    );
    Ok(())
}
fn error(error: impl std::fmt::Display) -> FeaRunError {
    FeaRunError::Observation(error.to_string())
}
