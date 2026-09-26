use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use runmat_analysis_execution::{
    prepare_planned_resource_graph_usage, ImportedAccuracy, ImportedPublishedResult,
    ImportedSemanticState, ImportedStructuralModalResult, ImportedStructuralRandomResponseResult,
    PlannedResourceUsageGraph, PreparedResourceUsage, StructuralRandomResponseStudyCoordinator,
    StructuralRandomResponseStudyPreparation,
};

use crate::{
    analysis_task_pool::{
        configure_local_analysis, execute_analysis_task_batch, planned_resource_attempt,
        AnalysisTaskBatchPolicy, AnalysisTaskDependency, AnalysisTaskPool, LocalAnalysisTaskPool,
        PreparedAnalysisTask, RemoteAnalysisTaskPool,
    },
    NativeExecutionConfig, NativeExecutionError, NativeExecutionResult, NativeProgramSession,
    RemotePoolDriver, StructuralSequenceExecutionPolicy,
};

#[derive(Clone, Debug, PartialEq)]
pub struct CompletedStructuralRandomResponseStudy {
    pub modal: ImportedStructuralModalResult,
    pub basis_state: ImportedSemanticState,
    pub resolved_workload: runmat_analysis_fea::execution::StructuralRandomResponseWorkload,
    pub response: ImportedStructuralRandomResponseResult,
    pub accuracy: ImportedAccuracy,
    pub resource_usage: PreparedResourceUsage,
    pub publication: ImportedPublishedResult,
    pub statistics_state: ImportedSemanticState,
}

pub async fn execute_structural_random_response_locally(
    mut config: NativeExecutionConfig,
    preparation: StructuralRandomResponseStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralRandomResponseStudy> {
    policy.validate()?;
    configure_local_analysis(&mut config, &policy.resources);
    let session = NativeProgramSession::new(config)?;
    execute_with_pool(
        &mut LocalAnalysisTaskPool::new(session),
        preparation,
        policy,
    )
    .await
}

pub async fn execute_structural_random_response_remotely(
    pool: Arc<RemotePoolDriver>,
    preparation: StructuralRandomResponseStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralRandomResponseStudy> {
    execute_with_pool(&mut RemoteAnalysisTaskPool::new(pool), preparation, policy).await
}

async fn execute_with_pool(
    pool: &mut impl AnalysisTaskPool,
    preparation: StructuralRandomResponseStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralRandomResponseStudy> {
    policy.validate()?;
    let publication_access = preparation.artifact_access.clone();
    let chunk_bytes = preparation.chunk_bytes;
    let inventory_limits = preparation.inventory_limits;
    let mut coordinator =
        StructuralRandomResponseStudyCoordinator::new(preparation).map_err(protocol)?;

    let modal = coordinator.prepare_modal().map_err(protocol)?;
    let [executed_modal] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host: &modal.host,
            program: &modal.program,
            resource_plan: Some(&modal.resource_plan),
            fallback_resources: policy.resources.clone(),
            dependencies: Vec::new(),
        }],
        task_policy(&policy),
    )
    .await?
    .try_into()
    .map_err(|_| protocol("modal extraction returned an invalid task inventory"))?;
    let modal_dependency = dependency(&executed_modal);
    coordinator
        .complete_modal(pool, executed_modal.response, &modal)
        .map_err(protocol)?;

    let basis = coordinator.prepare_basis_state().map_err(protocol)?;
    let [executed_basis] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host: &basis.host,
            program: &basis.program,
            resource_plan: None,
            fallback_resources: policy.resources.clone(),
            dependencies: vec![modal_dependency],
        }],
        task_policy(&policy),
    )
    .await?
    .try_into()
    .map_err(|_| protocol("modal-basis publication returned an invalid task inventory"))?;
    let basis_dependency = dependency(&executed_basis);
    coordinator
        .complete_basis_state(pool, executed_basis.response, &basis)
        .map_err(protocol)?;

    let response_started = Instant::now();
    let intervals = coordinator.prepare_intervals().map_err(protocol)?;
    let prepared = intervals
        .iter()
        .map(|interval| PreparedAnalysisTask {
            host: &interval.host,
            program: &interval.program,
            resource_plan: Some(&interval.resource_plan),
            fallback_resources: policy.resources.clone(),
            dependencies: vec![basis_dependency],
        })
        .collect();
    let executed_intervals =
        execute_analysis_task_batch(pool, prepared, task_policy(&policy)).await?;
    let interval_dependencies = executed_intervals
        .iter()
        .map(dependency)
        .collect::<Vec<_>>();
    let mut resource_attempts = Vec::with_capacity(intervals.len() + 1);
    let mut responses = BTreeMap::new();
    for (interval, executed) in intervals.iter().zip(executed_intervals) {
        resource_attempts.push(planned_resource_attempt(
            &interval.resource_plan,
            &executed,
        )?);
        responses.insert(interval.host.workload.identity.digest(), executed.response);
    }
    coordinator
        .complete_intervals(pool, intervals, responses)
        .map_err(protocol)?;

    let assembly = coordinator.prepare_assembly().map_err(protocol)?;
    let [executed_assembly] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host: &assembly.host,
            program: &assembly.program,
            resource_plan: Some(&assembly.resource_plan),
            fallback_resources: policy.resources.clone(),
            dependencies: interval_dependencies,
        }],
        task_policy(&policy),
    )
    .await?
    .try_into()
    .map_err(|_| protocol("random-response assembly returned an invalid task inventory"))?;
    let assembly_dependency = dependency(&executed_assembly);
    resource_attempts.push(planned_resource_attempt(
        &assembly.resource_plan,
        &executed_assembly,
    )?);
    coordinator
        .complete_assembly(pool, executed_assembly.response, &assembly)
        .map_err(protocol)?;
    let graph = PlannedResourceUsageGraph::new(
        resource_attempts,
        u64::try_from(response_started.elapsed().as_millis()).unwrap_or(u64::MAX),
    )
    .map_err(protocol)?;
    let resource_usage = prepare_planned_resource_graph_usage(
        &assembly.resource_plan,
        coordinator
            .resolved_workload()
            .ok_or_else(|| protocol("resolved random-response workload is absent"))?
            .identity()
            .digest(),
        coordinator
            .result()
            .ok_or_else(|| protocol("completed random-response result is absent"))?
            .identity
            .digest(),
        &graph,
        &publication_access,
        chunk_bytes,
        inventory_limits,
    )
    .map_err(protocol)?;
    pool.register_result_objects(&resource_usage.objects, &publication_access)?;

    let accuracy = coordinator.prepare_accuracy().map_err(protocol)?;
    let [executed_accuracy] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host: &accuracy.host,
            program: &accuracy.program,
            resource_plan: None,
            fallback_resources: policy.resources.clone(),
            dependencies: vec![assembly_dependency],
        }],
        task_policy(&policy),
    )
    .await?
    .try_into()
    .map_err(|_| protocol("random-response accuracy returned an invalid task inventory"))?;
    let accuracy_dependency = dependency(&executed_accuracy);
    coordinator
        .complete_accuracy(pool, executed_accuracy.response, &accuracy)
        .map_err(protocol)?;

    let publication = coordinator
        .prepare_publication(resource_usage)
        .map_err(protocol)?;
    let [executed_publication] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host: &publication.host,
            program: &publication.program,
            resource_plan: None,
            fallback_resources: policy.resources.clone(),
            dependencies: vec![accuracy_dependency],
        }],
        task_policy(&policy),
    )
    .await?
    .try_into()
    .map_err(|_| protocol("random-response publication returned an invalid task inventory"))?;
    let publication_dependency = dependency(&executed_publication);
    coordinator
        .complete_publication(pool, executed_publication.response, &publication)
        .map_err(protocol)?;

    let statistics = coordinator.prepare_statistics_state().map_err(protocol)?;
    let [executed_statistics] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host: &statistics.host,
            program: &statistics.program,
            resource_plan: None,
            fallback_resources: policy.resources.clone(),
            dependencies: vec![publication_dependency],
        }],
        task_policy(&policy),
    )
    .await?
    .try_into()
    .map_err(|_| protocol("random-statistics publication returned an invalid task inventory"))?;
    coordinator
        .complete_statistics_state(pool, executed_statistics.response, &statistics)
        .map_err(protocol)?;

    Ok(CompletedStructuralRandomResponseStudy {
        modal: coordinator
            .modal_result()
            .cloned()
            .ok_or_else(|| protocol("completed modal result is absent"))?,
        basis_state: coordinator
            .basis_state()
            .cloned()
            .ok_or_else(|| protocol("published modal-basis state is absent"))?,
        resolved_workload: coordinator
            .resolved_workload()
            .cloned()
            .ok_or_else(|| protocol("resolved random-response workload is absent"))?,
        response: coordinator
            .result()
            .cloned()
            .ok_or_else(|| protocol("completed random-response result is absent"))?,
        accuracy: coordinator
            .accuracy()
            .cloned()
            .ok_or_else(|| protocol("random-response accuracy is absent"))?,
        resource_usage: coordinator
            .resource_usage()
            .cloned()
            .ok_or_else(|| protocol("random-response resource usage is absent"))?,
        publication: coordinator
            .publication()
            .cloned()
            .ok_or_else(|| protocol("random-response publication is absent"))?,
        statistics_state: coordinator
            .statistics_state()
            .cloned()
            .ok_or_else(|| protocol("published random-statistics state is absent"))?,
    })
}

fn dependency(
    executed: &crate::analysis_task_pool::ExecutedAnalysisTask,
) -> AnalysisTaskDependency {
    AnalysisTaskDependency {
        identity: executed.identity,
        task_id: executed.task_id,
    }
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
