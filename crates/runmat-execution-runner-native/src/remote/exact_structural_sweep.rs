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
    let dynamic_studies = decoded
        .studies
        .iter()
        .all(|study| study.recipe.execution_mode == crate::STRUCTURAL_STUDY_ROOT_EXECUTION_MODE);
    if dynamic_studies {
        return execute_structural_studies(pool, request, decoded, run_id, resources).await;
    }
    if decoded.studies.iter().any(|study| {
        study.recipe.execution_mode != crate::EXACT_STRUCTURAL_STUDY_ROOT_EXECUTION_MODE
    }) {
        return Err(protocol(
            "structural sweep cannot mix pre-meshed and geometry-derived study roots",
        ));
    }
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
    let all_input_plans = studies
        .iter()
        .flat_map(|(_, study, _)| study.input_materializations.iter().cloned())
        .collect::<Vec<_>>();
    let input_execution = super::input_materialization::execute_ready(
        Arc::clone(&pool),
        &all_input_plans,
        &decoded.artifact_access,
        &resources,
    );
    let meshing_execution =
        execute_shared_meshing(Arc::clone(&pool), meshing_requests, &run_id, &resources);
    let (input_outcomes, meshing_outcomes) = tokio::join!(input_execution, meshing_execution);
    let mut input_outcomes = input_outcomes?;
    let meshing_outcomes = meshing_outcomes?;

    if decoded.composition.fail_fast() {
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

    // Mesh-dependent plans cannot be deduplicated or submitted before their mesh succeeds.
    for (group, meshing) in meshing_outcomes.iter().enumerate() {
        let SharedMeshingOutcome::Completed(completed) = meshing else {
            continue;
        };
        let group_plans = studies
            .iter()
            .filter(|(_, _, meshing_index)| *meshing_index == group)
            .flat_map(|(_, study, _)| study.input_materializations.iter().cloned())
            .collect::<Vec<_>>();
        input_outcomes.extend(
            super::input_materialization::execute_deferred(
                Arc::clone(&pool),
                &group_plans,
                &completed.result.artifact,
                &decoded.artifact_access,
                &resources,
            )
            .await?,
        );
    }

    let mut tasks = tokio::task::JoinSet::new();
    for (study_index, (study, study_root, meshing_index)) in studies.into_iter().enumerate() {
        let shared_meshing = meshing_outcomes[meshing_index].clone();
        let pool = Arc::clone(&pool);
        let resources = resources.clone();
        let input_outcomes = input_outcomes.clone();
        tasks.spawn(async move {
            let outcome = match shared_meshing {
                SharedMeshingOutcome::Completed(completed) => {
                    super::exact_structural_study::execute_with_completed_meshing(
                        pool,
                        &study,
                        study_root,
                        completed,
                        &input_outcomes,
                        &resources,
                        super::exact_structural_study::StructuralTransientRecoveryAdmission::Unavailable,
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
                if decoded.composition.fail_fast() && !cancelled_siblings {
                    pool.cancel(CancellationReason::ParentScope)?;
                    cancelled_siblings = true;
                }
                ExactStructuralSweepRemoteOutcome::Failed(message)
            }
            RemoteExactStructuralStudyComposition::Cancelled => {
                ExactStructuralSweepRemoteOutcome::Cancelled
            }
            RemoteExactStructuralStudyComposition::Indeterminate(message) => {
                if decoded.composition.fail_fast() && !cancelled_siblings {
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

async fn execute_structural_studies(
    pool: Arc<RemotePoolDriver>,
    request: ProgramExecutionRequest,
    decoded: crate::DecodedExactStructuralSweepRoot,
    run_id: String,
    resources: AllocationResources,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    let mut tasks = tokio::task::JoinSet::new();
    for (study_index, study) in decoded.studies.into_iter().enumerate() {
        let pool = Arc::clone(&pool);
        let resources = resources.clone();
        let task_scope = format!("{run_id}/study/{study_index}");
        tasks.spawn(async move {
            let mut recovery = crate::RetainedStructuralTransientRecovery::default();
            let outcome = super::structural_study::execute_for_composition(
                pool,
                &study,
                &task_scope,
                &resources,
                None,
                &mut recovery,
            )
            .await?;
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
                if decoded.composition.fail_fast() && !cancelled_siblings {
                    pool.cancel(CancellationReason::ParentScope)?;
                    cancelled_siblings = true;
                }
                ExactStructuralSweepRemoteOutcome::Failed(message)
            }
            RemoteExactStructuralStudyComposition::Cancelled => {
                ExactStructuralSweepRemoteOutcome::Cancelled
            }
            RemoteExactStructuralStudyComposition::Indeterminate(message) => {
                if decoded.composition.fail_fast() && !cancelled_siblings {
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

async fn execute_shared_meshing(
    pool: Arc<RemotePoolDriver>,
    meshing_requests: Vec<ProgramExecutionRequest>,
    run_id: &str,
    resources: &AllocationResources,
) -> NativeExecutionResult<Vec<SharedMeshingOutcome>> {
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
                        *terminal,
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
    Ok(meshing_outcomes)
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
    packaged(response, pool).map(Ok)
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
