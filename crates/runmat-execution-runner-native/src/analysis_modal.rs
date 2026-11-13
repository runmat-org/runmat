use std::sync::Arc;

use runmat_analysis_execution::{
    import_structural_modal_result, prepare_structural_modal_task, ImportedStructuralModalResult,
    StructuralModalTaskPreparation,
};

use crate::{
    analysis_task_pool::{
        configure_local_analysis, execute_analysis_task_batch, AnalysisTaskBatchPolicy,
        AnalysisTaskPool, LocalAnalysisTaskPool, PreparedAnalysisTask, RemoteAnalysisTaskPool,
    },
    NativeExecutionConfig, NativeExecutionError, NativeExecutionResult, NativeProgramSession,
    RemotePoolDriver, StructuralSequenceExecutionPolicy,
};

#[derive(Clone, Debug, PartialEq)]
pub struct CompletedStructuralModalStudy {
    pub result: ImportedStructuralModalResult,
}

pub async fn execute_structural_modal_locally(
    mut config: NativeExecutionConfig,
    preparation: StructuralModalTaskPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralModalStudy> {
    configure_local_analysis(&mut config, &policy.resources);
    let session = NativeProgramSession::new(config)?;
    execute_with_pool(
        &mut LocalAnalysisTaskPool::new(session),
        preparation,
        policy,
    )
    .await
}

pub async fn execute_structural_modal_remotely(
    pool: Arc<RemotePoolDriver>,
    preparation: StructuralModalTaskPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralModalStudy> {
    execute_with_pool(&mut RemoteAnalysisTaskPool::new(pool), preparation, policy).await
}

async fn execute_with_pool(
    pool: &mut impl AnalysisTaskPool,
    preparation: StructuralModalTaskPreparation,
    policy: StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralModalStudy> {
    policy.validate()?;
    let limits = preparation.inventory_limits;
    let prepared = prepare_structural_modal_task(preparation).map_err(|error| match error {
        runmat_analysis_execution::AnalysisBridgeError::Resource(error) => {
            NativeExecutionError::Resource(error)
        }
        error => protocol(error),
    })?;
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
    .map_err(|_| protocol("modal extraction returned an invalid task inventory"))?;
    let result = import_structural_modal_result(pool, executed.response, &prepared, limits)
        .map_err(protocol)?;
    if runmat_analysis_fea::execution::AnalysisTaskIdentity::new(
        result.host.workload.identity,
        Vec::new(),
    )
    .map_err(protocol)?
        != executed.identity
    {
        return Err(protocol(
            "modal result importer and task substrate derived different identities",
        ));
    }
    Ok(CompletedStructuralModalStudy { result })
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
