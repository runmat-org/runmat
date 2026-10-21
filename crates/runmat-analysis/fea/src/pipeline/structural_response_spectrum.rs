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
        StructuralModalResponseSpectrumDirectionRequest,
        StructuralModalResponseSpectrumDirectionResult, StructuralModalResponseSpectrumScope,
        StructuralModalResponseSpectrumScopeInput, StructuralModalResponseSpectrumWorkload,
    },
    planning::resource::{
        ResourcePlanningError, StructuralModalResponseSpectrumResourceProjection,
    },
    solve::dynamics::{
        recover_modal_response_spectrum_fields,
        solve_modal_response_spectrum_direction_with_operator, ResponseSpectrumFieldRecovery,
    },
};

/// FEA-owned physical preparation shared by local and portable response-spectrum execution.
///
/// The response-spectrum step consumes a modal basis, but its operator is the operator of the
/// basis-producing modal step. Keeping that reconstruction here prevents schedulers and native
/// runners from independently inventing producer selection, assembly, or identity rules.
pub struct PreparedStructuralModalResponseSpectrum {
    scope: StructuralModalResponseSpectrumScope,
    summary: AssemblySummary,
    modal_state: CanonicalSemanticState,
    resources: StructuralModalResponseSpectrumResourceProjection,
}

impl PreparedStructuralModalResponseSpectrum {
    pub fn scope(&self) -> &StructuralModalResponseSpectrumScope {
        &self.scope
    }

    pub fn resources(&self) -> &StructuralModalResponseSpectrumResourceProjection {
        &self.resources
    }
}

pub fn prepare_structural_modal_response_spectrum(
    workload: &StructuralModalResponseSpectrumWorkload,
    solver_mesh: &SolverMeshArtifact,
    modal_state: &CanonicalSemanticState,
    constraints: &CanonicalStructuralConstraintPlan,
    accelerator_requests: &[runmat_execution::resource::AcceleratorRequest],
) -> AnalysisExecutionResult<PreparedStructuralModalResponseSpectrum> {
    workload.validate()?;
    modal_state.validate()?;
    solver_mesh
        .validate_canonical()
        .map_err(|error| invalid(error.to_string()))?;
    if solver_mesh.geometry.geometry_revision != u64::from(workload.model().geometry_revision) {
        return Err(invalid(
            "response-spectrum workload and solver mesh geometry revisions differ",
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
            "response-spectrum modal state and topology transition differ",
        ));
    }
    let basis = modal_state.payload.modal_basis().ok_or_else(|| {
        invalid("response-spectrum preparation requires a modal-basis semantic state")
    })?;
    let producer = workload
        .model()
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id == transition.producer_step_id)
        .ok_or(AnalysisExecutionError::Identity(
            "response-spectrum modal-basis producer is absent",
        ))?;
    if producer.family != PhysicsFamily::Structural
        || producer.procedure != AnalysisProcedure::ModalExtraction
        || producer.solution_space != SolutionSpace::Direct
    {
        return Err(invalid(
            "response-spectrum basis producer must be a structural modal-extraction step",
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
            "response-spectrum basis was extracted from another structural operator",
        ));
    }
    let scope = StructuralModalResponseSpectrumScope::new(
        workload,
        StructuralModalResponseSpectrumScopeInput {
            topology_digest,
            source_state: modal_state.identity,
            basis: basis.identity(),
            operator_digest,
            physical_dof_count: summary.dof_count,
        },
    )?;
    let resources = StructuralModalResponseSpectrumResourceProjection::new(
        solver_mesh,
        constraints,
        workload,
        modal_state,
        accelerator_requests,
    )
    .map_err(resource_error)?;
    Ok(PreparedStructuralModalResponseSpectrum {
        scope,
        summary,
        modal_state: modal_state.clone(),
        resources,
    })
}

pub fn execute_structural_modal_response_spectrum_direction(
    workload: &StructuralModalResponseSpectrumWorkload,
    prepared: &PreparedStructuralModalResponseSpectrum,
    request: &StructuralModalResponseSpectrumDirectionRequest,
) -> AnalysisExecutionResult<StructuralModalResponseSpectrumDirectionResult> {
    request.validate_for(workload)?;
    if request.scope() != &prepared.scope {
        return Err(AnalysisExecutionError::Identity(
            "response-spectrum direction request names another prepared physical scope",
        ));
    }
    let basis = prepared
        .modal_state
        .payload
        .modal_basis()
        .ok_or_else(|| invalid("prepared response-spectrum modal basis is absent"))?;
    let numerical = solve_modal_response_spectrum_direction_with_operator(
        &prepared.summary,
        basis,
        request.spectrum(),
    )
    .map_err(|error| invalid(error.to_string()))?;
    StructuralModalResponseSpectrumDirectionResult::new(request.clone(), numerical)
}

/// Recovers statistically combined linear fields from the exact prepared producer operator.
///
/// Recovery deliberately consumes the signed modal contributions retained by the numerical
/// result. The terminal statistical DOF envelope is never treated as a simultaneous deformation.
pub fn recover_structural_modal_response_spectrum_fields(
    workload: &StructuralModalResponseSpectrumWorkload,
    prepared: &PreparedStructuralModalResponseSpectrum,
    result: &crate::solve::dynamics::ModalResponseSpectrumResult,
) -> AnalysisExecutionResult<ResponseSpectrumFieldRecovery> {
    prepared.scope.validate_for(workload)?;
    let basis = prepared
        .modal_state
        .payload
        .modal_basis()
        .ok_or_else(|| invalid("prepared response-spectrum modal basis is absent"))?;
    recover_modal_response_spectrum_fields(&prepared.summary, basis, result)
        .map_err(|error| invalid(error.to_string()))
}

fn resource_error(error: ResourcePlanningError) -> AnalysisExecutionError {
    invalid(error.to_string())
}

fn invalid(message: impl Into<String>) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(message.into())
}
