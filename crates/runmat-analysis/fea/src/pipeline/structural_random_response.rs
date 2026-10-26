use runmat_analysis_core::{AnalysisProcedure, PhysicsFamily, SolutionSpace};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    assembly::{
        assemble_structural_system_with_constraint_plan, identity::structural_operator_digest,
        AssemblySummary, StructuralFormulationRequirement,
    },
    constraints::CanonicalStructuralConstraintPlan,
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    execution::{
        AnalysisExecutionError, AnalysisExecutionResult, CanonicalSemanticState,
        StructuralRandomResponseIntervalRequest, StructuralRandomResponseIntervalResult,
        StructuralRandomResponseScope, StructuralRandomResponseScopeInput,
        StructuralRandomResponseWorkload,
    },
    planning::resource::{ResourcePlanningError, StructuralRandomResponseResourceProjection},
    solve::dynamics::{describe_modal_random_response, solve_modal_random_response_interval},
};

/// FEA-owned physical preparation shared by local and portable random-response execution.
///
/// The compact scope is transportable; the assembled operator and basis remain here for interval
/// workers that actually evaluate the frequency response. Deterministic fan-in needs only the
/// scope's canonical plan description and interval results.
pub struct PreparedStructuralRandomResponse {
    scope: StructuralRandomResponseScope,
    summary: AssemblySummary,
    modal_state: CanonicalSemanticState,
    resources: StructuralRandomResponseResourceProjection,
}

impl PreparedStructuralRandomResponse {
    pub fn scope(&self) -> &StructuralRandomResponseScope {
        &self.scope
    }

    pub fn resources(&self) -> &StructuralRandomResponseResourceProjection {
        &self.resources
    }
}

pub fn prepare_structural_random_response(
    workload: &StructuralRandomResponseWorkload,
    solver_mesh: &SolverMeshArtifact,
    modal_state: &CanonicalSemanticState,
    constraints: &CanonicalStructuralConstraintPlan,
) -> AnalysisExecutionResult<PreparedStructuralRandomResponse> {
    workload.validate()?;
    modal_state.validate()?;
    solver_mesh
        .validate_canonical()
        .map_err(|error| invalid(error.to_string()))?;
    if solver_mesh.geometry.geometry_revision != u64::from(workload.model().geometry_revision) {
        return Err(invalid(
            "random-response workload and solver mesh geometry revisions differ",
        ));
    }
    let topology_digest = Digest::from_bytes(*solver_mesh.canonical_digest.bytes());
    let transition = workload.modal_basis_transition();
    if modal_state.identity != transition.source_state
        || modal_state.topology_digest != topology_digest
        || transition.source_topology_digest != topology_digest
        || transition.target_topology_digest != topology_digest
    {
        return Err(AnalysisExecutionError::Identity(
            "random-response modal state and topology transition differ",
        ));
    }
    let basis = modal_state.payload.modal_basis().ok_or_else(|| {
        invalid("random-response preparation requires a modal-basis semantic state")
    })?;
    let producer = workload
        .model()
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id == transition.producer_step_id)
        .ok_or(AnalysisExecutionError::Identity(
            "random-response modal-basis producer is absent",
        ))?;
    if producer.family != PhysicsFamily::Structural
        || producer.procedure != AnalysisProcedure::ModalExtraction
        || producer.solution_space != SolutionSpace::Direct
    {
        return Err(invalid(
            "random-response basis producer must be a structural modal-extraction step",
        ));
    }
    let excitations =
        realize_step_excitations(workload.model(), producer, ExcitationSamplePoint::Static)
            .map_err(|error| invalid(error.to_string()))?;
    let summary = assemble_structural_system_with_constraint_plan(
        workload.model(),
        &excitations,
        Some(solver_mesh.clone()),
        StructuralFormulationRequirement::SyntheticOperator,
        constraints,
    )
    .map_err(|error| invalid(error.to_string()))?;
    super::admission::admit_structural(&summary, producer)
        .map_err(|error| invalid(error.to_string()))?;
    super::validate_rotational_dof_targets(workload.model(), &summary)
        .map_err(|error| invalid(error.to_string()))?;
    basis
        .validate_for(&summary)
        .map_err(|error| invalid(error.to_string()))?;
    let operator_digest =
        structural_operator_digest(&summary).map_err(|error| invalid(error.to_string()))?;
    if basis.operator_digest() != operator_digest {
        return Err(AnalysisExecutionError::Identity(
            "random-response basis was extracted from another structural operator",
        ));
    }
    let excitation = workload.excitation()?;
    let plan =
        describe_modal_random_response(&summary, basis, excitation, workload.integration_policy())
            .map_err(|error| invalid(error.to_string()))?;
    let scope = StructuralRandomResponseScope::new(
        workload,
        StructuralRandomResponseScopeInput {
            topology_digest,
            source_state: modal_state.identity,
            operator_digest,
            plan,
        },
    )?;
    let resources = StructuralRandomResponseResourceProjection::new(
        solver_mesh,
        constraints,
        workload,
        modal_state,
    )
    .map_err(resource_error)?;
    Ok(PreparedStructuralRandomResponse {
        scope,
        summary,
        modal_state: modal_state.clone(),
        resources,
    })
}

fn resource_error(error: ResourcePlanningError) -> AnalysisExecutionError {
    invalid(error.to_string())
}

pub fn execute_structural_random_response_intervals(
    workload: &StructuralRandomResponseWorkload,
    prepared: &PreparedStructuralRandomResponse,
    request: &StructuralRandomResponseIntervalRequest,
) -> AnalysisExecutionResult<StructuralRandomResponseIntervalResult> {
    request.validate_for(workload)?;
    if request.scope() != &prepared.scope {
        return Err(AnalysisExecutionError::Identity(
            "random-response interval request names another prepared physical scope",
        ));
    }
    let basis = prepared
        .modal_state
        .payload
        .modal_basis()
        .ok_or_else(|| invalid("prepared random-response modal basis is absent"))?;
    let excitation = workload.excitation()?;
    let range = request.interval_range();
    let intervals = (range[0]..range[1])
        .map(|ordinal| {
            solve_modal_random_response_interval(
                &prepared.summary,
                basis,
                excitation,
                workload.integration_policy(),
                ordinal,
            )
            .map_err(|error| invalid(error.to_string()))
        })
        .collect::<AnalysisExecutionResult<Vec<_>>>()?;
    StructuralRandomResponseIntervalResult::new(request.clone(), intervals)
}

fn invalid(message: impl Into<String>) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(message.into())
}
