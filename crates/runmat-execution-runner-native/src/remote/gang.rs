use std::collections::BTreeSet;
use std::sync::Arc;

use runmat_execution::identity::ArtifactId;
use runmat_execution::resource::ResourceRequest;
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::value::ValuePayload;
use runmat_execution::{
    CancellationReason, GangHandle, OutputContract, ProgramCallable, ProgramInvocationContext,
    SpmdTaskContext, TaskId,
};
use runmat_execution_artifact::{
    ProgramArtifact, ProgramBuildRecipe, ProgramExecutionDescriptor, ProgramExecutionInputs,
    ProgramExecutionRequest, PROGRAM_EXECUTION_REQUEST_SCHEMA_V5,
};
use runmat_execution_runner::{AttemptSuccess, DriverCommand, TaskSubmission};
use runmat_types::{LabRank, ParallelRegionId};

use super::pool_progress::RemoteTaskCompletion;
use super::RemotePoolDriver;
use crate::{NativeExecutionError, NativeExecutionResult, NativeProgramFailure};

/// One compiler-bound SPMD region submitted to a remote pool as an exact gang.
/// The recipe and artifact are immutable across ranks; only the typed rank
/// context and scheduler assignment differ.
#[derive(Clone, Debug)]
pub struct RemoteSpmdGangProgram {
    pub gang: GangHandle,
    pub region: ParallelRegionId,
    pub recipe: ProgramBuildRecipe,
    pub artifact: ProgramArtifact,
    pub captures: Vec<ValuePayload>,
    pub requested_outputs: u16,
    pub resources: ResourceRequest,
}

pub struct RemoteSpmdGangCompletion {
    pool: Arc<RemotePoolDriver>,
    gang: GangHandle,
    requested_outputs: usize,
    ranks: Vec<(LabRank, RemoteTaskCompletion)>,
}

impl RemotePoolDriver {
    pub fn submit_spmd_gang(
        self: &Arc<Self>,
        program: RemoteSpmdGangProgram,
    ) -> NativeExecutionResult<RemoteSpmdGangCompletion> {
        validate_gang_program(self, &program)?;
        let mut submissions = Vec::with_capacity(program.gang.labs.0 as usize);
        let mut programs = Vec::with_capacity(program.gang.labs.0 as usize);
        let mut receivers = Vec::with_capacity(program.gang.labs.0 as usize);
        let callable = ProgramCallable::spmd_region(program.region);
        let artifact_id = ArtifactId::derive(&[program.artifact.id.0.bytes()]);

        for rank_number in 1..=program.gang.labs.0 {
            let rank = LabRank(rank_number);
            let context = ProgramInvocationContext::SpmdTask {
                task: SpmdTaskContext {
                    gang: program.gang.clone(),
                    region: program.region,
                    rank,
                },
            };
            let request = ProgramExecutionRequest::from_parts(
                ProgramExecutionDescriptor {
                    schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V5,
                    recipe: program.recipe.clone(),
                    artifact: program.artifact.clone(),
                    callable: callable.clone(),
                    requested_outputs: program.requested_outputs,
                },
                ProgramExecutionInputs {
                    schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V5,
                    context: context.clone(),
                    arguments: program.captures.clone(),
                },
            )
            .map_err(protocol)?;
            request.validate_for_portable_host().map_err(protocol)?;
            let task_id = TaskId::derive(&[
                program.gang.id.bytes(),
                &program.gang.generation.to_be_bytes(),
                &rank.0.to_be_bytes(),
            ]);
            submissions.push(TaskSubmission {
                request: TaskRequest {
                    id: task_id,
                    scope_id: program.gang.scope_id,
                    pool_id: program.gang.pool.id,
                    program_artifact_id: artifact_id,
                    callable: Callable::for_program("remote-spmd", &callable),
                    invocation_context: context,
                    inputs: program.captures.clone(),
                    outputs: OutputContract {
                        requested_outputs: program.requested_outputs,
                    },
                    resources: program.resources.clone(),
                    retry: RetryPolicy::Never,
                    deadline_unix_millis: None,
                },
                dependencies: BTreeSet::new(),
                priority: 0,
            });
            programs.push((task_id, request));
            let (sender, receiver) = tokio::sync::oneshot::channel();
            receivers.push((task_id, rank, sender, receiver));
        }

        let actions = self
            .driver
            .lock()
            .expect("remote driver poisoned")
            .handle(DriverCommand::SubmitBatch(submissions))?;
        {
            let mut catalog = self
                .programs
                .lock()
                .expect("remote program catalog poisoned");
            for (task_id, request) in programs {
                catalog.insert(task_id, request);
            }
        }
        let mut ranks = Vec::with_capacity(receivers.len());
        for (task_id, rank, sender, receiver) in receivers {
            let progress = Arc::new(super::pool_progress::RemoteProgressBuffer::default());
            self.completions
                .lock()
                .expect("remote completion registry poisoned")
                .insert(task_id, sender);
            self.progress
                .lock()
                .expect("remote progress registry poisoned")
                .insert(task_id, Arc::clone(&progress));
            ranks.push((rank, RemoteTaskCompletion::new(receiver, progress)));
        }
        self.dispatch(actions);
        Ok(RemoteSpmdGangCompletion {
            pool: Arc::clone(self),
            gang: program.gang,
            requested_outputs: usize::from(program.requested_outputs),
            ranks,
        })
    }

    pub(super) fn cancel_spmd_gang(
        self: &Arc<Self>,
        gang: &GangHandle,
        reason: CancellationReason,
    ) -> NativeExecutionResult<()> {
        self.collectives.fail_gang(gang, "SPMD gang was cancelled");
        let actions = self.driver.lock().expect("remote driver poisoned").handle(
            DriverCommand::CancelGang {
                gang: gang.clone(),
                reason,
                now_millis: super::pool::now_millis(),
            },
        )?;
        self.dispatch(actions);
        Ok(())
    }
}

impl RemoteSpmdGangCompletion {
    pub async fn wait(
        self,
    ) -> Result<Vec<runmat_runtime::execution::SpmdRankResult>, NativeProgramFailure> {
        let mut pending = tokio::task::JoinSet::new();
        for (rank, completion) in self.ranks {
            pending.spawn(async move { (rank, completion.wait_ordered().await) });
        }
        let mut results = vec![None; self.gang.labs.0 as usize];
        let mut first_failure: Option<(u64, NativeProgramFailure)> = None;
        let mut cancellation_sent = false;
        while let Some(joined) = pending.join_next().await {
            let (rank, (order, result)) = joined.map_err(|error| {
                NativeProgramFailure::Infrastructure(format!(
                    "remote SPMD completion task stopped: {error}"
                ))
            })?;
            match result {
                Ok(AttemptSuccess::Spmd { outputs }) if outputs.len() == self.requested_outputs => {
                    results[rank.0 as usize - 1] =
                        Some(runmat_runtime::execution::SpmdRankResult { rank, outputs });
                }
                Ok(AttemptSuccess::Spmd { .. }) => retain_first_failure(
                    &mut first_failure,
                    order,
                    NativeProgramFailure::Infrastructure(
                        "remote SPMD rank returned the wrong output cardinality".into(),
                    ),
                ),
                Ok(AttemptSuccess::Values { .. }) => retain_first_failure(
                    &mut first_failure,
                    order,
                    NativeProgramFailure::Infrastructure(
                        "remote SPMD rank returned an ordinary task result".into(),
                    ),
                ),
                Err(failure) => retain_first_failure(&mut first_failure, order, failure),
            }
            if first_failure.is_some() && !cancellation_sent {
                self.pool
                    .cancel_spmd_gang(&self.gang, CancellationReason::DependencyFailed)
                    .map_err(|error| NativeProgramFailure::Infrastructure(error.to_string()))?;
                cancellation_sent = true;
            }
        }
        if let Some((_, failure)) = first_failure {
            return Err(failure);
        }
        results
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                NativeProgramFailure::Infrastructure(
                    "remote SPMD gang completed without every admitted rank".into(),
                )
            })
    }
}

fn validate_gang_program(
    pool: &RemotePoolDriver,
    program: &RemoteSpmdGangProgram,
) -> NativeExecutionResult<()> {
    program.gang.validate().map_err(protocol)?;
    program.resources.validate().map_err(protocol)?;
    if program.gang.scope_id != pool.scope_id
        || program.gang.pool.id != pool.pool_id
        || program.gang.labs.0 == 0
    {
        return Err(NativeExecutionError::Configuration(
            "remote SPMD gang is outside this pool authority".into(),
        ));
    }
    let workers = pool
        .driver
        .lock()
        .expect("remote driver poisoned")
        .snapshot()
        .pools
        .get(&pool.pool_id)
        .map(|pool| pool.workers.len())
        .unwrap_or_default();
    if workers < program.gang.labs.0 as usize {
        return Err(NativeExecutionError::Configuration(
            "remote SPMD gang exceeds the registered worker count".into(),
        ));
    }
    Ok(())
}

fn retain_first_failure(
    first: &mut Option<(u64, NativeProgramFailure)>,
    order: u64,
    failure: NativeProgramFailure,
) {
    if first.as_ref().is_none_or(|(current, _)| order < *current) {
        *first = Some((order, failure));
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
