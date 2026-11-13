//! Native placement of original J2 blocks through the shared coordinator and task pool.
use crate::{
    analysis_task_pool::{
        execute_governed_analysis_task, AnalysisTaskBatchPolicy, AnalysisTaskDependency,
        AnalysisTaskPool, LocalAnalysisTaskPool, PreparedAnalysisTask,
    },
    NativeExecutionError, NativeExecutionResult, NativeProgramSession,
};
use runmat_analysis_execution::{
    CompletedNonlinearSolidStudy, NonlinearSolidStudyCoordinator, NonlinearSolidStudyPhase,
    NonlinearSolidStudyPreparation, PreparedAnalysisTaskClosure,
};
use runmat_execution::{resource::ResourceRequest, task::RetryPolicy};

pub(crate) mod source;

#[derive(Clone, Debug)]
pub struct NonlinearSolidStudyExecutionPolicy {
    pub resources: ResourceRequest,
    pub retry: RetryPolicy,
    pub deadline_unix_millis: Option<u64>,
    pub priority: i16,
}

/// The session owns normal native scheduling, cancellation and worker termination.
/// Retain `session.object_source()` before passing the session to preserve cold results.
pub async fn execute_nonlinear_solid_study_locally(
    session: NativeProgramSession,
    preparation: NonlinearSolidStudyPreparation,
    policy: NonlinearSolidStudyExecutionPolicy,
) -> NativeExecutionResult<CompletedNonlinearSolidStudy> {
    execute_with_pool(
        &mut LocalAnalysisTaskPool::new(session),
        preparation,
        policy,
    )
    .await
}

pub(crate) async fn execute_with_pool(
    pool: &mut impl AnalysisTaskPool,
    preparation: NonlinearSolidStudyPreparation,
    policy: NonlinearSolidStudyExecutionPolicy,
) -> NativeExecutionResult<CompletedNonlinearSolidStudy> {
    policy.resources.validate().map_err(protocol)?;
    let scope = pool.scope_id();
    let placement = pool.pool_id();
    let output_policy = preparation.output_policy;
    let mut coordinator = NonlinearSolidStudyCoordinator::new(preparation).map_err(protocol)?;
    let mut predecessor = None;
    macro_rules! execute {
        ($task:expr) => {{
            let task = $task;
            let executed = execute_governed_analysis_task(
                pool,
                PreparedAnalysisTask {
                    host: &task.host,
                    program: &task.program,
                    resource_plan: Some(&task.resource_plan),
                    prepared_closure: Some(PreparedAnalysisTaskClosure {
                        inputs: &task.inputs,
                        output_policy,
                    }),
                    dependencies: predecessor.into_iter().collect(),
                    fallback_resources: policy.resources.clone(),
                },
                AnalysisTaskBatchPolicy {
                    retry: policy.retry,
                    deadline_unix_millis: policy.deadline_unix_millis,
                    priority: policy.priority,
                },
            )
            .await?;
            predecessor = Some(AnalysisTaskDependency {
                identity: executed.0.identity,
                task_id: executed.0.task_id,
            });
            executed
        }};
    }
    loop {
        if pool.scope_id() != scope || pool.pool_id() != placement {
            return Err(protocol(
                "nonlinear lifecycle pool/scope changed between original phases",
            ));
        }
        match coordinator.phase() {
            NonlinearSolidStudyPhase::ReadyForBlock { .. } => {
                let (executed, accounting) =
                    execute!(coordinator.prepare_block().map_err(protocol)?);
                coordinator
                    .complete_block(pool, executed.response, accounting)
                    .map_err(protocol)?;
            }
            NonlinearSolidStudyPhase::ReadyForFieldRecovery => {
                let (executed, accounting) =
                    execute!(coordinator.prepare_field_recovery().map_err(protocol)?);
                coordinator
                    .complete_field_recovery(pool, executed.response, accounting)
                    .map_err(protocol)?;
            }
            NonlinearSolidStudyPhase::ReadyForObservationFanout => {
                let count = coordinator
                    .prepare_observation_fanout(pool)
                    .map_err(protocol)?
                    .partitions
                    .len();
                for ordinal in 0..count {
                    let (executed, accounting) = execute!(coordinator
                        .observation_partition(ordinal)
                        .map_err(protocol)?);
                    coordinator
                        .complete_observation_partition(
                            pool,
                            ordinal,
                            executed.response,
                            accounting,
                        )
                        .map_err(protocol)?;
                }
            }
            NonlinearSolidStudyPhase::ReadyForObservationReduction { .. } => {
                let (executed, accounting) = execute!(coordinator
                    .prepare_observation_reduction()
                    .map_err(protocol)?);
                coordinator
                    .complete_observation_reduction(pool, executed.response, accounting)
                    .map_err(protocol)?;
            }
            NonlinearSolidStudyPhase::ReadyForObservationAssembly => {
                let (executed, accounting) = execute!(coordinator
                    .prepare_observation_assembly()
                    .map_err(protocol)?);
                coordinator
                    .complete_observation_assembly(pool, executed.response, accounting)
                    .map_err(protocol)?;
            }
            NonlinearSolidStudyPhase::ReadyForAccuracy => {
                let (executed, accounting) =
                    execute!(coordinator.prepare_accuracy().map_err(protocol)?);
                coordinator
                    .complete_accuracy(pool, executed.response, accounting)
                    .map_err(protocol)?;
            }
            NonlinearSolidStudyPhase::ReadyForState { .. } => {
                let (executed, accounting) =
                    execute!(coordinator.prepare_next_state().map_err(protocol)?);
                coordinator
                    .complete_next_state(pool, executed.response, accounting)
                    .map_err(protocol)?;
            }
            NonlinearSolidStudyPhase::ReadyForBaseline => {
                coordinator.freeze_baseline().map_err(protocol)?;
            }
            NonlinearSolidStudyPhase::ReadyForPublication => {
                let (executed, accounting) =
                    execute!(coordinator.prepare_publication().map_err(protocol)?);
                coordinator
                    .complete_publication(pool, executed.response, accounting)
                    .map_err(protocol)?;
            }
            NonlinearSolidStudyPhase::Complete => return coordinator.finish().map_err(protocol),
            _ => {
                return Err(protocol(
                    "nonlinear lifecycle reached unexpected awaiting phase",
                ))
            }
        }
    }
}
fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
