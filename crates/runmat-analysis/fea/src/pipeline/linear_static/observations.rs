//! Post-solve observation admission and recovery on the original field owners.
use runmat_analysis_core::{AnalysisField, AnalysisModel, ProcedureStep};
use runmat_meshing_core::SolverMeshArtifact;
use runmat_simulation_usage::{LedgerError, OperationCount};

use crate::{
    algebra::NumericalCapacityPhase,
    connectors::CanonicalStructuralConnectorResult,
    constraints::{CanonicalStructuralConstraintPlan, CanonicalStructuralConstraintResult},
    contracts::FeaRunError,
    post::{
        fields::take_structural_observable_fields, recover_connector_observable_fields,
        recover_constraint_group_observable_fields,
    },
    results::{
        field::{RecoveredFieldCatalog, StructuralRecoveredFieldTopology},
        observation::{
            evaluate_structural_observations, resolve_structural_observation_plan_with_connectors,
            structural_observation_derivation_work_upper_bound, ObservationLimits,
            ObservationResult,
        },
    },
};

pub(super) struct StaticObservationInputs<'a> {
    pub model: &'a AnalysisModel,
    pub step: &'a ProcedureStep,
    pub mesh: Option<&'a SolverMeshArtifact>,
    pub constraints: &'a CanonicalStructuralConstraintPlan,
    pub constraint_result: &'a CanonicalStructuralConstraintResult,
    pub connector_result: &'a CanonicalStructuralConnectorResult,
}

/// Plan resolution and evaluation each derive one observation topology. Their
/// composed quote admits both before field ownership moves or work is recorded.
pub(crate) fn postsolve_work_upper_bound(
    mesh: Option<&SolverMeshArtifact>,
) -> Result<OperationCount, FeaRunError> {
    structural_observation_derivation_work_upper_bound(mesh)?
        .checked_mul(2)
        .map_err(|cause| FeaRunError::Usage(LedgerError::from(cause)))
}

pub(super) async fn evaluate<O, F, A>(
    input: StaticObservationInputs<'_>,
    fields: &mut Vec<AnalysisField>,
    before: Option<&mut O>,
) -> Result<Vec<ObservationResult>, A>
where
    O: FnMut(NumericalCapacityPhase, OperationCount) -> F,
    F: std::future::Future<Output = Result<(), A>>,
    A: From<FeaRunError>,
{
    if input.step.observations.is_empty() {
        return Ok(Vec::new());
    }
    if let Some(before) = before {
        let work = postsolve_work_upper_bound(input.mesh)?;
        if work > OperationCount::ZERO {
            before(NumericalCapacityPhase::StaticObservations, work).await?;
        }
    }
    let invalid_field = |error: crate::results::field::RecoveredFieldError| {
        FeaRunError::Observation(error.to_string())
    };
    let mut recovered = take_structural_observable_fields(fields).map_err(invalid_field)?;
    recovered.extend(
        recover_constraint_group_observable_fields(input.constraint_result)
            .map_err(invalid_field)?,
    );
    recovered.extend(
        recover_connector_observable_fields(input.connector_result).map_err(invalid_field)?,
    );
    let mut topology =
        StructuralRecoveredFieldTopology::derive(input.model, input.mesh).map_err(invalid_field)?;
    topology.entity_counts.constraint_groups = input.constraint_result.groups.len();
    topology.entity_counts.connectors = input.connector_result.responses.len();
    let catalog = RecoveredFieldCatalog::new_for_topology(
        topology.identity.discretization_digest(),
        topology.entity_counts,
        recovered,
    )
    .map_err(invalid_field)?;
    let definitions = catalog
        .fields()
        .iter()
        .map(|field| field.definition().clone())
        .collect::<Vec<_>>();
    let plan = resolve_structural_observation_plan_with_connectors(
        input.step,
        input.model,
        input.mesh,
        input.constraints,
        &input.connector_result.plan,
        &definitions,
        ObservationLimits::default(),
    )
    .map_err(FeaRunError::from)?;
    let observations = evaluate_structural_observations(&plan, input.model, input.mesh, &catalog)
        .map_err(FeaRunError::from)?;
    fields.extend(
        catalog
            .into_fields()
            .into_iter()
            .map(|field| field.into_storage()),
    );
    Ok(observations)
}
