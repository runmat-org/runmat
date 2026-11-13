//! Phase-local prepared tasks for preload, tangent state, modal extraction and basis publication.
use super::*;

pub(super) async fn preload(
    pool: &mut impl AnalysisTaskPool,
    coordinator: &mut StructuralPrestressedModalFrequencyStudyCoordinator,
    policy: &StructuralSequenceExecutionPolicy,
    dependencies: Vec<AnalysisTaskDependency>,
) -> NativeExecutionResult<AnalysisTaskDependency> {
    let preload = coordinator.prepare_preload().map_err(protocol)?;
    let executed_preload = execute_one(
        pool,
        &preload.host,
        &preload.program,
        &preload.resource_plan,
        policy,
        dependencies,
        "preload",
    )
    .await?;
    let preload_dependency = dependency(&executed_preload);
    coordinator
        .complete_preload(pool, executed_preload.response, &preload)
        .map_err(protocol)?;
    verify_identity(
        coordinator.preload_task_identity(),
        executed_preload.identity,
        "preload",
    )?;

    Ok(preload_dependency)
}

pub(super) async fn prestress(
    pool: &mut impl AnalysisTaskPool,
    coordinator: &mut StructuralPrestressedModalFrequencyStudyCoordinator,
    policy: &StructuralSequenceExecutionPolicy,
    dependencies: Vec<AnalysisTaskDependency>,
) -> NativeExecutionResult<(AnalysisTaskDependency, PlannedResourceUsageAttempt)> {
    let prestress = coordinator.prepare_prestress_state().map_err(protocol)?;
    let executed_prestress = execute_one(
        pool,
        &prestress.host,
        &prestress.program,
        &prestress.resource_plan,
        policy,
        dependencies,
        "prestress-state publication",
    )
    .await?;
    let prestress_usage = planned_resource_attempt(&prestress.resource_plan, &executed_prestress)?;
    let prestress_dependency = dependency(&executed_prestress);
    coordinator
        .complete_prestress_state(pool, executed_prestress.response, &prestress)
        .map_err(protocol)?;
    verify_identity(
        coordinator.prestress_state_task_identity(),
        executed_prestress.identity,
        "prestress-state publication",
    )?;

    Ok((prestress_dependency, prestress_usage))
}

pub(super) async fn modal(
    pool: &mut impl AnalysisTaskPool,
    coordinator: &mut StructuralPrestressedModalFrequencyStudyCoordinator,
    policy: &StructuralSequenceExecutionPolicy,
    dependencies: Vec<AnalysisTaskDependency>,
) -> NativeExecutionResult<AnalysisTaskDependency> {
    let modal = coordinator.prepare_modal().map_err(protocol)?;
    let executed_modal = execute_one(
        pool,
        &modal.host,
        &modal.program,
        &modal.resource_plan,
        policy,
        dependencies,
        "prestressed-modal solve",
    )
    .await?;
    let modal_dependency = dependency(&executed_modal);
    coordinator
        .complete_modal(pool, executed_modal.response, &modal)
        .map_err(protocol)?;
    verify_identity(
        coordinator.modal_task_identity(),
        executed_modal.identity,
        "prestressed-modal solve",
    )?;

    Ok(modal_dependency)
}

pub(super) async fn basis(
    pool: &mut impl AnalysisTaskPool,
    coordinator: &mut StructuralPrestressedModalFrequencyStudyCoordinator,
    policy: &StructuralSequenceExecutionPolicy,
    dependencies: Vec<AnalysisTaskDependency>,
) -> NativeExecutionResult<(AnalysisTaskDependency, PlannedResourceUsageAttempt)> {
    let basis = coordinator.prepare_basis_state().map_err(protocol)?;
    let executed_basis = execute_one(
        pool,
        &basis.host,
        &basis.program,
        &basis.resource_plan,
        policy,
        dependencies,
        "modal-basis publication",
    )
    .await?;
    let basis_usage = planned_resource_attempt(&basis.resource_plan, &executed_basis)?;
    let basis_dependency = dependency(&executed_basis);
    coordinator
        .complete_basis_state(pool, executed_basis.response, &basis)
        .map_err(protocol)?;
    verify_identity(
        coordinator.basis_state_task_identity(),
        executed_basis.identity,
        "modal-basis publication",
    )?;

    Ok((basis_dependency, basis_usage))
}
