use std::sync::Arc;

use runmat_analysis_execution::{
    import_structural_buckling_result, prepare_structural_buckling_task, ImportedSemanticState,
    ImportedStructuralBucklingResult, ImportedStructuralLinearResult, PlannedResourceUsageAttempt,
    StructuralBucklingStudyCoordinator, StructuralBucklingStudyPreparation,
    StructuralBucklingTaskPreparation,
};

use crate::{
    analysis_task_pool::{
        configure_local_analysis, execute_analysis_task_batch, planned_resource_attempt,
        AnalysisTaskBatchPolicy, AnalysisTaskPool, LocalAnalysisTaskPool, PreparedAnalysisTask,
        RemoteAnalysisTaskPool,
    },
    NativeExecutionConfig, NativeExecutionError, NativeExecutionResult, NativeProgramSession,
    RemotePoolDriver, StructuralSequenceExecutionPolicy,
};

#[derive(Clone, Debug, PartialEq)]
pub struct CompletedStructuralBucklingStudy {
    pub result: ImportedStructuralBucklingResult,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompletedStructuralBucklingSequenceStudy {
    pub preload: ImportedStructuralLinearResult,
    pub prestress_state: ImportedSemanticState,
    pub state_publication_usage: PlannedResourceUsageAttempt,
    pub buckling: ImportedStructuralBucklingResult,
}

pub async fn execute_structural_buckling_study_locally(
    mut config: NativeExecutionConfig,
    preparation: StructuralBucklingStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralBucklingSequenceStudy> {
    configure_local_analysis(&mut config, &policy.resources);
    let session = NativeProgramSession::new(config)?;
    execute_study_with_pool(
        &mut LocalAnalysisTaskPool::new(session),
        preparation,
        policy,
    )
    .await
}

pub async fn execute_structural_buckling_study_remotely(
    pool: Arc<RemotePoolDriver>,
    preparation: StructuralBucklingStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralBucklingSequenceStudy> {
    execute_study_with_pool(&mut RemoteAnalysisTaskPool::new(pool), preparation, policy).await
}

async fn execute_study_with_pool(
    pool: &mut impl AnalysisTaskPool,
    preparation: StructuralBucklingStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralBucklingSequenceStudy> {
    execute_with_thermal_source(pool, preparation, policy, None).await
}

pub(crate) async fn execute_with_thermal_source(
    pool: &mut impl AnalysisTaskPool,
    preparation: StructuralBucklingStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
    thermal: Option<crate::analysis_thermal_prestress::BoundThermalExecution>,
) -> NativeExecutionResult<CompletedStructuralBucklingSequenceStudy> {
    policy.validate()?;
    if preparation.accelerator_requests != policy.resources.accelerators {
        return Err(protocol(
            "buckling preparation and execution policy name different accelerator requests",
        ));
    }
    let (mut coordinator, preload_dependencies) = match thermal {
        Some(thermal) => {
            let (source, dependency) = thermal.into_parts(pool)?;
            (
                StructuralBucklingStudyCoordinator::new_with_thermal_source(preparation, source)
                    .map_err(protocol)?,
                vec![dependency],
            )
        }
        None => (
            StructuralBucklingStudyCoordinator::new(preparation).map_err(protocol)?,
            Vec::new(),
        ),
    };
    crate::analysis_thermal_prestress::validate_preload_dependencies(
        &coordinator.preload_dependencies(),
        &preload_dependencies,
    )?;

    let preload = coordinator.prepare_preload().map_err(protocol)?;
    let [executed_preload] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host: &preload.host,
            program: &preload.program,
            resource_plan: Some(&preload.resource_plan),
            prepared_closure: None,
            fallback_resources: policy.resources.clone(),
            dependencies: preload_dependencies,
        }],
        task_policy(&policy),
    )
    .await?
    .try_into()
    .map_err(|_| protocol("buckling preload returned an invalid task inventory"))?;
    coordinator
        .complete_preload(pool, executed_preload.response, &preload)
        .map_err(protocol)?;
    if coordinator.preload_task_identity() != Some(executed_preload.identity) {
        return Err(protocol(
            "buckling coordinator and preload task substrate derived different identities",
        ));
    }

    let state = coordinator.prepare_state().map_err(protocol)?;
    let [executed_state] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host: &state.host,
            program: &state.program,
            resource_plan: Some(&state.resource_plan),
            prepared_closure: None,
            fallback_resources: policy.resources.clone(),
            dependencies: vec![crate::analysis_task_pool::AnalysisTaskDependency {
                identity: executed_preload.identity,
                task_id: executed_preload.task_id,
            }],
        }],
        task_policy(&policy),
    )
    .await?
    .try_into()
    .map_err(|_| protocol("prestress publication returned an invalid task inventory"))?;
    let state_publication_usage = planned_resource_attempt(&state.resource_plan, &executed_state)?;
    coordinator
        .complete_state(pool, executed_state.response, &state)
        .map_err(protocol)?;
    if coordinator.state_task_identity() != Some(executed_state.identity) {
        return Err(protocol(
            "buckling coordinator and state task substrate derived different identities",
        ));
    }

    let buckling = coordinator.prepare_buckling().map_err(protocol)?;
    let [executed_buckling] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host: &buckling.host,
            program: &buckling.program,
            resource_plan: Some(&buckling.resource_plan),
            prepared_closure: None,
            fallback_resources: policy.resources.clone(),
            dependencies: vec![crate::analysis_task_pool::AnalysisTaskDependency {
                identity: executed_state.identity,
                task_id: executed_state.task_id,
            }],
        }],
        task_policy(&policy),
    )
    .await?
    .try_into()
    .map_err(|_| protocol("buckling solve returned an invalid task inventory"))?;
    coordinator
        .complete_buckling(pool, executed_buckling.response, &buckling)
        .map_err(protocol)?;
    if coordinator.buckling_task_identity() != Some(executed_buckling.identity) {
        return Err(protocol(
            "buckling coordinator and solve task substrate derived different identities",
        ));
    }
    Ok(CompletedStructuralBucklingSequenceStudy {
        preload: coordinator
            .preload_result()
            .cloned()
            .ok_or_else(|| protocol("completed buckling preload is absent"))?,
        prestress_state: coordinator
            .semantic_state()
            .cloned()
            .ok_or_else(|| protocol("published buckling prestress state is absent"))?,
        state_publication_usage,
        buckling: coordinator
            .buckling_result()
            .cloned()
            .ok_or_else(|| protocol("completed buckling result is absent"))?,
    })
}

pub async fn execute_structural_buckling_locally(
    mut config: NativeExecutionConfig,
    preparation: StructuralBucklingTaskPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralBucklingStudy> {
    configure_local_analysis(&mut config, &policy.resources);
    let session = NativeProgramSession::new(config)?;
    execute_with_pool(
        &mut LocalAnalysisTaskPool::new(session),
        preparation,
        policy,
    )
    .await
}

pub async fn execute_structural_buckling_remotely(
    pool: Arc<RemotePoolDriver>,
    preparation: StructuralBucklingTaskPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralBucklingStudy> {
    execute_with_pool(&mut RemoteAnalysisTaskPool::new(pool), preparation, policy).await
}

async fn execute_with_pool(
    pool: &mut impl AnalysisTaskPool,
    preparation: StructuralBucklingTaskPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralBucklingStudy> {
    policy.validate()?;
    let limits = preparation.inventory_limits;
    let prepared = prepare_structural_buckling_task(preparation).map_err(protocol)?;
    let [executed] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host: &prepared.host,
            program: &prepared.program,
            resource_plan: Some(&prepared.resource_plan),
            prepared_closure: None,
            fallback_resources: policy.resources,
            dependencies: Vec::new(),
        }],
        AnalysisTaskBatchPolicy {
            retry: policy.retry,
            deadline_unix_millis: policy.deadline_unix_millis,
            priority: policy.priority,
        },
    )
    .await?
    .try_into()
    .map_err(|_| protocol("buckling execution returned an invalid task inventory"))?;
    let result = import_structural_buckling_result(pool, executed.response, &prepared, limits)
        .map_err(protocol)?;
    if runmat_analysis_fea::execution::AnalysisTaskIdentity::new(
        result.host.workload.identity,
        Vec::new(),
    )
    .map_err(protocol)?
        != executed.identity
    {
        return Err(protocol(
            "buckling result importer and task substrate derived different identities",
        ));
    }
    Ok(CompletedStructuralBucklingStudy { result })
}

fn task_policy(policy: &StructuralSequenceExecutionPolicy) -> AnalysisTaskBatchPolicy {
    AnalysisTaskBatchPolicy {
        retry: policy.retry,
        deadline_unix_millis: policy.deadline_unix_millis,
        priority: policy.priority,
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
