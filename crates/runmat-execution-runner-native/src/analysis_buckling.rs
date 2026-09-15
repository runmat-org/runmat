use std::sync::Arc;

use runmat_analysis_execution::{
    import_structural_buckling_result, prepare_structural_buckling_task,
    ImportedStructuralBucklingResult, StructuralBucklingTaskPreparation,
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
pub struct CompletedStructuralBucklingStudy {
    pub result: ImportedStructuralBucklingResult,
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

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
