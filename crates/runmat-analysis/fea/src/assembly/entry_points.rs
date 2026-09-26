//! Owned compatibility and borrowed-mesh entry points for structural assembly.

use super::*;

/// Internal reduced operator used by non-structural experimental pipelines.
/// Structural callers use `assemble_structural_system` for explicit synthetic admission.
pub(crate) fn assemble_linear_system(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    solver_mesh: Option<SolverMeshArtifact>,
) -> Result<AssemblySummary, LinearAssemblyError> {
    assemble_with_mesh(
        model,
        excitations,
        solver_mesh.as_ref(),
        None,
        None,
        crate::planning::resource::LocalExecutionPlan::serial(),
        None,
    )
}

pub(crate) fn assemble_structural_system(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    solver_mesh: Option<SolverMeshArtifact>,
    requirement: StructuralFormulationRequirement,
) -> Result<AssemblySummary, LinearAssemblyError> {
    assemble_structural_system_borrowed(model, excitations, solver_mesh.as_ref(), requirement)
}

/// Reassemble a time-dependent load without duplicating the immutable canonical mesh.
pub(crate) fn assemble_structural_system_borrowed(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    solver_mesh: Option<&SolverMeshArtifact>,
    requirement: StructuralFormulationRequirement,
) -> Result<AssemblySummary, LinearAssemblyError> {
    assemble_with_mesh(
        model,
        excitations,
        solver_mesh,
        Some(requirement),
        None,
        crate::planning::resource::LocalExecutionPlan::serial(),
        None,
    )
}

pub(crate) fn assemble_structural_system_with_constraint_plan(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    solver_mesh: Option<SolverMeshArtifact>,
    requirement: StructuralFormulationRequirement,
    constraint_plan: &CanonicalStructuralConstraintPlan,
) -> Result<AssemblySummary, LinearAssemblyError> {
    assemble_structural_system_with_execution(
        model,
        excitations,
        solver_mesh,
        requirement,
        Some(constraint_plan),
        crate::planning::resource::LocalExecutionPlan::serial(),
        None,
    )
}

pub(crate) fn assemble_structural_system_with_execution(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    solver_mesh: Option<SolverMeshArtifact>,
    requirement: StructuralFormulationRequirement,
    constraint_plan: Option<&CanonicalStructuralConstraintPlan>,
    local_execution: crate::planning::resource::LocalExecutionPlan,
    solved_nodal_temperature_k: Option<&[f64]>,
) -> Result<AssemblySummary, LinearAssemblyError> {
    assemble_with_mesh(
        model,
        excitations,
        solver_mesh.as_ref(),
        Some(requirement),
        constraint_plan,
        local_execution,
        solved_nodal_temperature_k,
    )
}

fn assemble_with_mesh(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    solver_mesh: Option<&SolverMeshArtifact>,
    requirement: Option<StructuralFormulationRequirement>,
    constraint_plan: Option<&CanonicalStructuralConstraintPlan>,
    local_execution: crate::planning::resource::LocalExecutionPlan,
    solved_nodal_temperature_k: Option<&[f64]>,
) -> Result<AssemblySummary, LinearAssemblyError> {
    let (thermo_mechanical_context, electro_thermal_context) =
        crate::excitation::realize_coupling_contexts(model, excitations)
            .map_err(LinearAssemblyError::CouplingRealization)?;
    if solved_nodal_temperature_k.is_some()
        && excitations.iter().any(|excitation| {
            matches!(
                excitation.kind,
                crate::excitation::RealizedExcitationKind::TemperatureField(_)
            )
        })
    {
        return Err(LinearAssemblyError::CouplingRealization(
            "solved thermal state conflicts with an active prescribed temperature field".into(),
        ));
    }
    assemble_linear_system_impl(
        model,
        excitations,
        solver_mesh,
        thermo_mechanical_context.filter(|_| solved_nodal_temperature_k.is_none()),
        electro_thermal_context,
        AssemblyExecution {
            structural_requirement: requirement,
            supplied_constraint_plan: constraint_plan,
            local_execution,
            solved_nodal_temperature_k,
        },
    )
}
