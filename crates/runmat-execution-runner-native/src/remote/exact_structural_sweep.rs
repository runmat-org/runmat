//! Driver-owned composition of exact structural studies in one remote worker allocation.

use std::sync::Arc;

use runmat_execution::CancellationReason;
use runmat_execution_artifact::ProgramExecutionRequest;
use runmat_execution_transport_native::control::ResourceRequest as AllocationResources;

use super::exact_structural_study::RemoteExactStructuralStudyComposition;
use super::pool_execution::RemotePoolExecutionOutcome;
use super::pool_tasks::packaged;
use super::RemotePoolDriver;
use crate::exact_structural_sweep_root::package_exact_structural_sweep_result;
use crate::exact_structural_sweep_root::ExactStructuralSweepRemoteOutcome;
use crate::{NativeExecutionError, NativeExecutionResult};

#[derive(Clone)]
enum SharedMeshingOutcome {
    Completed(Box<super::exact_meshing::CompletedRemoteExactMeshing>),
    Terminal(RemoteExactStructuralStudyComposition),
}

pub(crate) async fn execute(
    pool: Arc<RemotePoolDriver>,
    request: ProgramExecutionRequest,
    run_id: String,
    resources: AllocationResources,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    let decoded = crate::decode_exact_structural_sweep_root(&request, pool.as_ref())?;
    let mut meshing_requests = Vec::new();
    let mut studies = Vec::with_capacity(decoded.studies.len());
    for study in decoded.studies {
        let study_root = crate::decode_exact_structural_study_root(&study, pool.as_ref())?;
        let meshing_index = meshing_requests
            .iter()
            .position(|request| request == &study_root.exact_meshing)
            .unwrap_or_else(|| {
                meshing_requests.push(study_root.exact_meshing.clone());
                meshing_requests.len() - 1
            });
        studies.push((study, study_root, meshing_index));
    }

    let mut meshing_tasks = tokio::task::JoinSet::new();
    for (meshing_index, meshing_request) in meshing_requests.into_iter().enumerate() {
        let pool = Arc::clone(&pool);
        let resources = resources.clone();
        let task_scope = format!("{run_id}/mesh/{meshing_index}");
        meshing_tasks.spawn(async move {
            let outcome = super::exact_meshing::execute_for_composition(
                pool,
                &meshing_request,
                &task_scope,
                &resources,
            )
            .await?;
            let outcome = match outcome {
                super::exact_meshing::RemoteExactMeshingComposition::Completed(completed) => {
                    SharedMeshingOutcome::Completed(completed)
                }
                super::exact_meshing::RemoteExactMeshingComposition::Terminal(terminal) => {
                    SharedMeshingOutcome::Terminal(super::exact_structural_study::composition(
                        terminal,
                    ))
                }
            };
            Ok::<_, NativeExecutionError>((meshing_index, outcome))
        });
    }

    let mut meshing_outcomes = (0..meshing_tasks.len()).map(|_| None).collect::<Vec<_>>();
    while let Some(completed) = meshing_tasks.join_next().await {
        let (meshing_index, outcome) = completed
            .map_err(|error| protocol(format!("remote meshing coordinator failed: {error}")))??;
        meshing_outcomes[meshing_index] = Some(outcome);
    }
    let meshing_outcomes = meshing_outcomes
        .into_iter()
        .map(|outcome| outcome.ok_or_else(|| protocol("remote meshing group did not terminate")))
        .collect::<NativeExecutionResult<Vec<_>>>()?;

    if decoded.fail_fast {
        if let Some((failed_group, terminal)) =
            meshing_outcomes
                .iter()
                .enumerate()
                .find_map(|(index, outcome)| match outcome {
                    SharedMeshingOutcome::Terminal(
                        terminal @ (RemoteExactStructuralStudyComposition::Failed(_)
                        | RemoteExactStructuralStudyComposition::Indeterminate(_)),
                    ) => Some((index, terminal.clone())),
                    _ => None,
                })
        {
            pool.cancel(CancellationReason::ParentScope)?;
            let mut reported = false;
            let outcomes = studies
                .iter()
                .map(|(_, _, group)| {
                    if *group == failed_group && !reported {
                        reported = true;
                        composition_terminal(terminal.clone())
                    } else {
                        ExactStructuralSweepRemoteOutcome::Cancelled
                    }
                })
                .collect();
            return package(
                request,
                decoded.identity,
                decoded.artifact_access,
                outcomes,
                pool,
            );
        }
    }

    let mut tasks = tokio::task::JoinSet::new();
    for (study_index, (study, study_root, meshing_index)) in studies.into_iter().enumerate() {
        let shared_meshing = meshing_outcomes[meshing_index].clone();
        let pool = Arc::clone(&pool);
        let resources = resources.clone();
        tasks.spawn(async move {
            let outcome = match shared_meshing {
                SharedMeshingOutcome::Completed(completed) => {
                    super::exact_structural_study::execute_with_completed_meshing(
                        pool, &study, study_root, completed, &resources,
                    )
                    .await?
                }
                SharedMeshingOutcome::Terminal(terminal) => terminal,
            };
            Ok::<_, NativeExecutionError>((study_index, outcome))
        });
    }

    let mut outcomes = (0..tasks.len()).map(|_| None).collect::<Vec<_>>();
    let mut cancelled_siblings = false;
    while let Some(completed) = tasks.join_next().await {
        let (study_index, outcome) = completed
            .map_err(|error| protocol(format!("remote sweep coordinator failed: {error}")))??;
        let terminal = match outcome {
            RemoteExactStructuralStudyComposition::Completed(response) => {
                ExactStructuralSweepRemoteOutcome::Succeeded(response)
            }
            RemoteExactStructuralStudyComposition::Failed(message) => {
                if decoded.fail_fast && !cancelled_siblings {
                    pool.cancel(CancellationReason::ParentScope)?;
                    cancelled_siblings = true;
                }
                ExactStructuralSweepRemoteOutcome::Failed(message)
            }
            RemoteExactStructuralStudyComposition::Cancelled => {
                ExactStructuralSweepRemoteOutcome::Cancelled
            }
            RemoteExactStructuralStudyComposition::Indeterminate(message) => {
                if decoded.fail_fast && !cancelled_siblings {
                    pool.cancel(CancellationReason::ParentScope)?;
                    cancelled_siblings = true;
                }
                ExactStructuralSweepRemoteOutcome::Indeterminate(message)
            }
        };
        outcomes[study_index] = Some(terminal);
    }
    let outcomes = outcomes
        .into_iter()
        .map(|outcome| outcome.ok_or_else(|| protocol("remote sweep entry did not terminate")))
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    package(
        request,
        decoded.identity,
        decoded.artifact_access,
        outcomes,
        pool,
    )
}

fn composition_terminal(
    terminal: RemoteExactStructuralStudyComposition,
) -> ExactStructuralSweepRemoteOutcome {
    match terminal {
        RemoteExactStructuralStudyComposition::Completed(response) => {
            ExactStructuralSweepRemoteOutcome::Succeeded(response)
        }
        RemoteExactStructuralStudyComposition::Failed(message) => {
            ExactStructuralSweepRemoteOutcome::Failed(message)
        }
        RemoteExactStructuralStudyComposition::Cancelled => {
            ExactStructuralSweepRemoteOutcome::Cancelled
        }
        RemoteExactStructuralStudyComposition::Indeterminate(message) => {
            ExactStructuralSweepRemoteOutcome::Indeterminate(message)
        }
    }
}

fn package(
    request: ProgramExecutionRequest,
    identity: runmat_analysis_fea::execution::SweepIdentity,
    artifact_access: runmat_execution_artifact::ArtifactAccess,
    outcomes: Vec<ExactStructuralSweepRemoteOutcome>,
    pool: Arc<RemotePoolDriver>,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    let (response, objects) =
        package_exact_structural_sweep_result(&request, identity, &artifact_access, outcomes)?;
    for object in objects {
        pool.register_execution_object(object.reference, object.bytes)?;
    }
    packaged(response, pool).map(RemotePoolExecutionOutcome::Completed)
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
