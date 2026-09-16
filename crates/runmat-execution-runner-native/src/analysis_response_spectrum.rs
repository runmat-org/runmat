use std::collections::BTreeMap;
use std::sync::Arc;

use runmat_analysis_execution::{
    ImportedSemanticState, ImportedStructuralModalResponseSpectrumResult,
    ImportedStructuralModalResult, StructuralModalResponseSpectrumStudyCoordinator,
    StructuralModalResponseSpectrumStudyPreparation,
};

use crate::{
    analysis_task_pool::{
        configure_local_analysis, execute_analysis_task_batch, AnalysisTaskBatchPolicy,
        AnalysisTaskDependency, AnalysisTaskPool, LocalAnalysisTaskPool, PreparedAnalysisTask,
        RemoteAnalysisTaskPool,
    },
    NativeExecutionConfig, NativeExecutionError, NativeExecutionResult, NativeProgramSession,
    RemotePoolDriver, StructuralSequenceExecutionPolicy,
};

#[derive(Clone, Debug, PartialEq)]
pub struct CompletedStructuralModalResponseSpectrumStudy {
    pub modal: ImportedStructuralModalResult,
    pub basis_state: ImportedSemanticState,
    pub resolved_workload: runmat_analysis_fea::execution::StructuralModalResponseSpectrumWorkload,
    pub response: ImportedStructuralModalResponseSpectrumResult,
    pub envelope_state: ImportedSemanticState,
}

pub async fn execute_structural_modal_response_spectrum_locally(
    mut config: NativeExecutionConfig,
    preparation: StructuralModalResponseSpectrumStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralModalResponseSpectrumStudy> {
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

pub async fn execute_structural_modal_response_spectrum_remotely(
    pool: Arc<RemotePoolDriver>,
    preparation: StructuralModalResponseSpectrumStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralModalResponseSpectrumStudy> {
    execute_with_pool(&mut RemoteAnalysisTaskPool::new(pool), preparation, policy).await
}

async fn execute_with_pool(
    pool: &mut impl AnalysisTaskPool,
    preparation: StructuralModalResponseSpectrumStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralModalResponseSpectrumStudy> {
    policy.validate()?;
    let mut coordinator =
        StructuralModalResponseSpectrumStudyCoordinator::new(preparation).map_err(protocol)?;

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

    let directions = coordinator.prepare_directions().map_err(protocol)?;
    let prepared = directions
        .iter()
        .map(|direction| PreparedAnalysisTask {
            host: &direction.host,
            program: &direction.program,
            resource_plan: Some(&direction.resource_plan),
            fallback_resources: policy.resources.clone(),
            dependencies: vec![basis_dependency.clone()],
        })
        .collect();
    let executed_directions =
        execute_analysis_task_batch(pool, prepared, task_policy(&policy)).await?;
    let direction_dependencies = executed_directions
        .iter()
        .map(dependency)
        .collect::<Vec<_>>();
    let responses = directions
        .iter()
        .zip(executed_directions)
        .map(|(direction, executed)| (direction.host.workload.identity.digest(), executed.response))
        .collect::<BTreeMap<_, _>>();
    coordinator
        .complete_directions(pool, directions, responses)
        .map_err(protocol)?;

    let assembly = coordinator.prepare_assembly().map_err(protocol)?;
    let [executed_assembly] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host: &assembly.host,
            program: &assembly.program,
            resource_plan: Some(&assembly.resource_plan),
            fallback_resources: policy.resources.clone(),
            dependencies: direction_dependencies,
        }],
        task_policy(&policy),
    )
    .await?
    .try_into()
    .map_err(|_| protocol("response-spectrum assembly returned an invalid task inventory"))?;
    let assembly_dependency = dependency(&executed_assembly);
    coordinator
        .complete_assembly(pool, executed_assembly.response, &assembly)
        .map_err(protocol)?;

    let envelope = coordinator.prepare_envelope_state().map_err(protocol)?;
    let [executed_envelope] = execute_analysis_task_batch(
        pool,
        vec![PreparedAnalysisTask {
            host: &envelope.host,
            program: &envelope.program,
            resource_plan: None,
            fallback_resources: policy.resources.clone(),
            dependencies: vec![assembly_dependency],
        }],
        task_policy(&policy),
    )
    .await?
    .try_into()
    .map_err(|_| protocol("response-envelope publication returned an invalid task inventory"))?;
    coordinator
        .complete_envelope_state(pool, executed_envelope.response, &envelope)
        .map_err(protocol)?;

    Ok(CompletedStructuralModalResponseSpectrumStudy {
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
            .ok_or_else(|| protocol("resolved response-spectrum workload is absent"))?,
        response: coordinator
            .result()
            .cloned()
            .ok_or_else(|| protocol("completed response-spectrum result is absent"))?,
        envelope_state: coordinator
            .envelope_state()
            .cloned()
            .ok_or_else(|| protocol("published response-envelope state is absent"))?,
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
