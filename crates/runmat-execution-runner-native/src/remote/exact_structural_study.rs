//! Driver-owned whole-study composition for exact CAD and structural-static analysis.

use std::collections::BTreeMap;
use std::sync::Arc;

use runmat_analysis_execution::{
    StructuralSequencePreparation, StructuralStudyWorkloads, StructuralTransientRecoveryClosure,
    StructuralTransientSequencePreparationInput,
};
use runmat_analysis_fea::execution::{StructuralFrequencySolverPlan, DEFAULT_ANALYSIS_CHUNK_BYTES};
use runmat_execution::state::TaskState;
use runmat_execution::task::RetryPolicy;
use runmat_execution::Digest;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ProgramExecutionRequest, ProgramExecutionResponse};
use runmat_execution_transport_native::control::ResourceRequest as AllocationResources;

use super::exact_meshing::RemoteExactMeshingComposition;
use super::pool_execution::RemotePoolExecutionOutcome;
use super::pool_tasks::{execution_failure_outcome, packaged};
use super::RemotePoolDriver;
use crate::StructuralTransientRecoveryPublisher;
use crate::{BoundaryRegionResolution, NativeExecutionError, NativeExecutionResult};

#[derive(Clone, Debug)]
pub(super) enum RemoteExactStructuralStudyComposition {
    Completed(ProgramExecutionResponse),
    Failed(String),
    Cancelled,
    Indeterminate(String),
}

pub(super) enum StructuralTransientRecoveryAdmission<'a> {
    Unavailable,
    Available {
        restored: Option<&'a super::recovery::RestoredObjectClosure>,
        publisher: &'a mut dyn StructuralTransientRecoveryPublisher,
    },
}

pub(super) async fn execute(
    pool: Arc<RemotePoolDriver>,
    request: ProgramExecutionRequest,
    run_id: String,
    resources: AllocationResources,
    recovered: Option<&super::recovery::RestoredObjectClosure>,
    recovery_publisher: &mut super::recovery::RemoteRecoveryPublisher<'_>,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    match execute_for_composition(
        Arc::clone(&pool),
        &request,
        &run_id,
        &resources,
        recovered,
        Some(recovery_publisher),
    )
    .await?
    {
        RemoteExactStructuralStudyComposition::Completed(response) => {
            packaged(response, pool).map(RemotePoolExecutionOutcome::Completed)
        }
        RemoteExactStructuralStudyComposition::Failed(message) => {
            execution_failure_outcome(pool, message)
        }
        RemoteExactStructuralStudyComposition::Cancelled => {
            Ok(RemotePoolExecutionOutcome::Cancelled)
        }
        RemoteExactStructuralStudyComposition::Indeterminate(message) => {
            Ok(RemotePoolExecutionOutcome::Indeterminate(message))
        }
    }
}

pub(super) async fn execute_for_composition(
    pool: Arc<RemotePoolDriver>,
    request: &ProgramExecutionRequest,
    run_id: &str,
    resources: &AllocationResources,
    recovered: Option<&super::recovery::RestoredObjectClosure>,
    recovery_publisher: Option<&mut dyn StructuralTransientRecoveryPublisher>,
) -> NativeExecutionResult<RemoteExactStructuralStudyComposition> {
    let decoded = crate::decode_exact_structural_study_root(request, pool.as_ref())?;
    let signal_materializations = decoded.signal_materializations.clone();
    let artifact_access = decoded.artifact_access.clone();
    let meshing = super::exact_meshing::execute_for_composition(
        Arc::clone(&pool),
        &decoded.exact_meshing,
        run_id,
        resources,
    );
    let signals = super::signal_materialization::execute(
        Arc::clone(&pool),
        signal_materializations,
        &artifact_access,
        resources,
    );
    let (meshing, signal_outcomes) = tokio::join!(meshing, signals);
    let completed = match meshing? {
        RemoteExactMeshingComposition::Completed(completed) => completed,
        RemoteExactMeshingComposition::Terminal(outcome) => return Ok(composition(outcome)),
    };
    let recovery = match recovery_publisher {
        Some(publisher) => StructuralTransientRecoveryAdmission::Available {
            restored: recovered,
            publisher,
        },
        None if recovered.is_none() => StructuralTransientRecoveryAdmission::Unavailable,
        None => {
            return Err(protocol(
                "restored structural transient state requires a recovery publisher",
            ))
        }
    };
    Box::pin(execute_with_completed_meshing(
        pool,
        request,
        decoded,
        completed,
        &signal_outcomes?,
        resources,
        recovery,
    ))
    .await
}

#[cfg(all(test, feature = "occt-native"))]
pub(crate) async fn execute_for_test(
    pool: Arc<RemotePoolDriver>,
    request: &ProgramExecutionRequest,
    run_id: &str,
    resources: &AllocationResources,
    publisher: &mut dyn StructuralTransientRecoveryPublisher,
) -> NativeExecutionResult<Result<ProgramExecutionResponse, String>> {
    Ok(
        match execute_for_composition(pool, request, run_id, resources, None, Some(publisher))
            .await?
        {
            RemoteExactStructuralStudyComposition::Completed(response) => Ok(response),
            RemoteExactStructuralStudyComposition::Failed(message)
            | RemoteExactStructuralStudyComposition::Indeterminate(message) => Err(message),
            RemoteExactStructuralStudyComposition::Cancelled => {
                Err("exact structural study was cancelled".into())
            }
        },
    )
}

pub(super) async fn execute_with_completed_meshing(
    pool: Arc<RemotePoolDriver>,
    request: &ProgramExecutionRequest,
    decoded: crate::DecodedExactStructuralStudyRoot,
    completed: Box<super::exact_meshing::CompletedRemoteExactMeshing>,
    signal_outcomes: &BTreeMap<
        runmat_analysis_fea::execution::SignalMaterializationIdentity,
        super::signal_materialization::RemoteSignalMaterializationOutcome,
    >,
    resources: &AllocationResources,
    recovery: StructuralTransientRecoveryAdmission<'_>,
) -> NativeExecutionResult<RemoteExactStructuralStudyComposition> {
    if completed.result.artifact.geometry.source_digest.bytes()
        != decoded.geometry_source_digest.bytes()
    {
        return Ok(RemoteExactStructuralStudyComposition::Failed(
            "meshed geometry differs from the whole-study source identity".into(),
        ));
    }

    let aliases = completed
        .source_face_ids
        .iter()
        .map(|(index, identity)| {
            (
                format!("face_{:06}", index.saturating_add(1)),
                identity.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let resolution = BoundaryRegionResolution {
        schema_version: 1,
        geometry_source_digest: decoded.geometry_source_digest,
        solver_mesh_digest: Digest::from_bytes(*completed.result.artifact.canonical_digest.bytes()),
        aliases,
    };
    resolution.validate()?;
    let boundary_objects = crate::exact_structural_study_root::prepare_boundary_resolution_objects(
        &resolution,
        &decoded.artifact_access,
        DEFAULT_ANALYSIS_CHUNK_BYTES,
        ObjectInventoryLimits::default(),
    )?;

    let unresolved_workloads_digest =
        crate::exact_structural_study_root::structural_workload_inventory_digest(
            &decoded.workloads,
        )?;
    let mut workloads = decoded.workloads.clone();
    let mut resolved_model = workloads.authored_model().map_err(protocol)?.clone();
    let completed_signals = match super::signal_materialization::apply_to_model(
        &mut resolved_model,
        &decoded.signal_materializations,
        signal_outcomes,
    ) {
        Ok(completed) => completed,
        Err(terminal) => return Ok(signal_composition(terminal)),
    };
    if let Err(error) = resolve_model_boundary_aliases(&mut resolved_model, &resolution) {
        return Ok(RemoteExactStructuralStudyComposition::Failed(error));
    }
    workloads.replace_model(resolved_model).map_err(protocol)?;
    let task_resources =
        match admit_analysis_task_resources(resources, &decoded.analysis_accelerator_requests) {
            Ok(resources) => resources,
            Err(error) => return Ok(RemoteExactStructuralStudyComposition::Failed(error)),
        };
    let analysis_accelerator_requests = decoded.analysis_accelerator_requests.clone();
    let policy = crate::StructuralSequenceExecutionPolicy {
        resources: task_resources,
        retry: RetryPolicy::ExplicitlyIdempotent {
            max_attempts: runmat_analysis_execution::DEFAULT_ANALYSIS_MAXIMUM_ATTEMPTS,
        },
        deadline_unix_millis: Some(super::pool_resources::task_deadline_unix_millis(resources)?),
        priority: 0,
    };
    let (response, objects) = match workloads {
        StructuralStudyWorkloads::HeterogeneousSequence { workloads } => {
            let authored = StructuralStudyWorkloads::HeterogeneousSequence { workloads };
            let step_ids = authored.step_ids().map_err(protocol)?;
            if let [preload_step, modal_step, frequency_step] = step_ids.as_slice() {
                let is_modal_frequency = match &authored {
                    StructuralStudyWorkloads::HeterogeneousSequence { workloads } => matches!(
                        (
                            workloads.get(preload_step),
                            workloads.get(modal_step),
                            workloads.get(frequency_step),
                        ),
                        (
                            Some(runmat_analysis_fea::execution::StructuralStepWorkload::Linear { .. }),
                            Some(runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedModal { .. }),
                            Some(runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedFrequency { .. }),
                        )
                    ),
                    _ => false,
                };
                if !is_modal_frequency {
                    return Ok(RemoteExactStructuralStudyComposition::Failed(
                        "three-step structural execution is not a prestressed modal-frequency sequence"
                            .into(),
                    ));
                }
                if matches!(
                    &recovery,
                    StructuralTransientRecoveryAdmission::Available {
                        restored: Some(_),
                        ..
                    }
                ) {
                    return Ok(RemoteExactStructuralStudyComposition::Failed(
                        "prestressed modal-frequency execution cannot consume a transient recovery closure"
                            .into(),
                    ));
                }
                let completion = match Box::pin(
                    crate::execute_structural_prestressed_modal_frequency_study_remotely(
                        Arc::clone(&pool),
                        runmat_analysis_execution::StructuralPrestressedFrequencyStudyPreparation {
                            authored_study_digest: decoded.authored_study_digest,
                            frozen_project_digest: decoded.frozen_project_digest,
                            workloads: authored,
                            mesh: completed.result.artifact.clone(),
                            artifact_access: decoded.artifact_access.clone(),
                            program_revision: request.recipe.program_revision.clone(),
                            chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
                            inventory_limits: ObjectInventoryLimits::default(),
                            required_capabilities: Default::default(),
                            accelerator_requests: analysis_accelerator_requests.clone(),
                        },
                        policy,
                    ),
                )
                .await
                {
                    Ok(completion) => completion,
                    Err(error) => return Ok(sequence_error_composition(&pool, error.to_string())),
                };
                let (response, objects) =
                    crate::exact_structural_study_root::package_exact_structural_study_result(
                        request,
                        &decoded,
                        &completed.result,
                        &boundary_objects,
                        crate::exact_structural_study_root::ExactStructuralAnalysisPublication::StructuralPrestressedModalFrequency {
                            signals: &completed_signals,
                            completion: &completion,
                            unresolved_workloads_digest,
                        },
                    )?;
                for object in objects {
                    pool.register_execution_object(object.reference, object.bytes)?;
                }
                return Ok(RemoteExactStructuralStudyComposition::Completed(response));
            }
            let [producer_step, consumer_step] = step_ids.as_slice() else {
                return Ok(RemoteExactStructuralStudyComposition::Failed(
                    "heterogeneous structural execution requires exactly two steps".into(),
                ));
            };
            let producer_is_linear = match &authored {
                StructuralStudyWorkloads::HeterogeneousSequence { workloads } => matches!(
                    workloads.get(producer_step),
                    Some(runmat_analysis_fea::execution::StructuralStepWorkload::Linear { .. })
                ),
                _ => false,
            };
            let consumer_is_prestressed_modal = match &authored {
                StructuralStudyWorkloads::HeterogeneousSequence { workloads } => matches!(
                    workloads.get(consumer_step),
                    Some(
                        runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedModal { .. }
                    )
                ),
                _ => false,
            };
            let consumer_is_prestressed_frequency = match &authored {
                StructuralStudyWorkloads::HeterogeneousSequence { workloads } => matches!(
                    workloads.get(consumer_step),
                    Some(
                        runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedFrequency {
                            ..
                        }
                    )
                ),
                _ => false,
            };
            if producer_is_linear {
                if matches!(
                    &recovery,
                    StructuralTransientRecoveryAdmission::Available {
                        restored: Some(_),
                        ..
                    }
                ) {
                    return Ok(RemoteExactStructuralStudyComposition::Failed(
                        "prestress execution cannot consume a transient recovery closure".into(),
                    ));
                }
                if consumer_is_prestressed_frequency {
                    let completion = match Box::pin(
                        crate::execute_structural_prestressed_frequency_study_remotely(
                            Arc::clone(&pool),
                            runmat_analysis_execution::StructuralPrestressedFrequencyStudyPreparation {
                                authored_study_digest: decoded.authored_study_digest,
                                frozen_project_digest: decoded.frozen_project_digest,
                                workloads: authored,
                                mesh: completed.result.artifact.clone(),
                                artifact_access: decoded.artifact_access.clone(),
                                program_revision: request.recipe.program_revision.clone(),
                                chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
                                inventory_limits: ObjectInventoryLimits::default(),
                                required_capabilities: Default::default(),
                                accelerator_requests: analysis_accelerator_requests.clone(),
                            },
                            policy,
                        ),
                    )
                    .await
                    {
                        Ok(completion) => completion,
                        Err(error) => {
                            return Ok(sequence_error_composition(&pool, error.to_string()))
                        }
                    };
                    let (response, objects) =
                        crate::exact_structural_study_root::package_exact_structural_study_result(
                            request,
                            &decoded,
                            &completed.result,
                            &boundary_objects,
                            crate::exact_structural_study_root::ExactStructuralAnalysisPublication::StructuralPrestressedFrequency {
                                signals: &completed_signals,
                                completion: &completion,
                                unresolved_workloads_digest,
                            },
                        )?;
                    for object in objects {
                        pool.register_execution_object(object.reference, object.bytes)?;
                    }
                    return Ok(RemoteExactStructuralStudyComposition::Completed(response));
                }
                if consumer_is_prestressed_modal {
                    let completion =
                        match Box::pin(crate::execute_structural_prestressed_modal_study_remotely(
                            Arc::clone(&pool),
                            runmat_analysis_execution::StructuralPrestressedModalStudyPreparation {
                                authored_study_digest: decoded.authored_study_digest,
                                frozen_project_digest: decoded.frozen_project_digest,
                                workloads: authored,
                                mesh: completed.result.artifact.clone(),
                                artifact_access: decoded.artifact_access.clone(),
                                program_revision: request.recipe.program_revision.clone(),
                                chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
                                inventory_limits: ObjectInventoryLimits::default(),
                                required_capabilities: Default::default(),
                                accelerator_requests: analysis_accelerator_requests.clone(),
                            },
                            policy,
                        ))
                        .await
                        {
                            Ok(completion) => completion,
                            Err(error) => {
                                return Ok(sequence_error_composition(&pool, error.to_string()))
                            }
                        };
                    let (response, objects) =
                        crate::exact_structural_study_root::package_exact_structural_study_result(
                            request,
                            &decoded,
                            &completed.result,
                            &boundary_objects,
                            crate::exact_structural_study_root::ExactStructuralAnalysisPublication::StructuralPrestressedModal {
                                signals: &completed_signals,
                                completion: &completion,
                                unresolved_workloads_digest,
                            },
                        )?;
                    for object in objects {
                        pool.register_execution_object(object.reference, object.bytes)?;
                    }
                    return Ok(RemoteExactStructuralStudyComposition::Completed(response));
                }
                let completion = match crate::execute_structural_buckling_study_remotely(
                    Arc::clone(&pool),
                    runmat_analysis_execution::StructuralBucklingStudyPreparation {
                        authored_study_digest: decoded.authored_study_digest,
                        frozen_project_digest: decoded.frozen_project_digest,
                        workloads: authored,
                        mesh: completed.result.artifact.clone(),
                        artifact_access: decoded.artifact_access.clone(),
                        program_revision: request.recipe.program_revision.clone(),
                        chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
                        inventory_limits: ObjectInventoryLimits::default(),
                        required_capabilities: Default::default(),
                        accelerator_requests: analysis_accelerator_requests.clone(),
                    },
                    policy,
                )
                .await
                {
                    Ok(completion) => completion,
                    Err(error) => return Ok(sequence_error_composition(&pool, error.to_string())),
                };
                let (response, objects) =
                    crate::exact_structural_study_root::package_exact_structural_study_result(
                    request,
                    &decoded,
                    &completed.result,
                    &boundary_objects,
                    crate::exact_structural_study_root::ExactStructuralAnalysisPublication::StructuralBuckling {
                        signals: &completed_signals,
                        completion: &completion,
                        unresolved_workloads_digest,
                    },
                )?;
                for object in objects {
                    pool.register_execution_object(object.reference, object.bytes)?;
                }
                return Ok(RemoteExactStructuralStudyComposition::Completed(response));
            }
            let (recovered, recovery_publisher) = match recovery {
                StructuralTransientRecoveryAdmission::Available {
                    restored,
                    publisher,
                } => (restored, publisher),
                StructuralTransientRecoveryAdmission::Unavailable => {
                    return Ok(RemoteExactStructuralStudyComposition::Failed(
                        "modal-transient execution requires a transactional recovery publisher"
                            .into(),
                    ))
                }
            };
            let recovery = recovered
                .map(|restored| {
                    StructuralTransientRecoveryClosure::from_objects(
                        &decoded.artifact_access,
                        restored.root.clone(),
                        restored.objects.clone(),
                    )
                    .map_err(protocol)
                })
                .transpose()?;
            let completion = match crate::execute_structural_modal_transient_remotely(
                Arc::clone(&pool),
                runmat_analysis_execution::StructuralModalTransientStudyPreparation {
                    authored_study_digest: decoded.authored_study_digest,
                    frozen_project_digest: decoded.frozen_project_digest,
                    workloads: authored,
                    mesh: completed.result.artifact.clone(),
                    artifact_access: decoded.artifact_access.clone(),
                    program_revision: request.recipe.program_revision.clone(),
                    chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
                    inventory_limits: ObjectInventoryLimits::default(),
                    required_capabilities: Default::default(),
                    accelerator_requests: analysis_accelerator_requests.clone(),
                },
                policy,
                recovery.as_ref(),
                recovery_publisher,
            )
            .await
            {
                Ok(completion) => completion,
                Err(error) => return Ok(sequence_error_composition(&pool, error.to_string())),
            };
            crate::exact_structural_study_root::package_exact_structural_study_result(
                request,
                &decoded,
                &completed.result,
                &boundary_objects,
                crate::exact_structural_study_root::ExactStructuralAnalysisPublication::StructuralModalTransient {
                    signals: &completed_signals,
                    completion: &completion,
                    unresolved_workloads_digest,
                },
            )?
        }
        StructuralStudyWorkloads::StructuralModal { workload } => {
            if matches!(
                &recovery,
                StructuralTransientRecoveryAdmission::Available {
                    restored: Some(_),
                    ..
                }
            ) {
                return Ok(RemoteExactStructuralStudyComposition::Failed(
                    "modal extraction cannot consume a transient recovery closure".into(),
                ));
            }
            let completion = match crate::execute_structural_modal_remotely(
                Arc::clone(&pool),
                runmat_analysis_execution::StructuralModalTaskPreparation {
                    authored_study_digest: decoded.authored_study_digest,
                    frozen_project_digest: decoded.frozen_project_digest,
                    program_revision: request.recipe.program_revision.clone(),
                    artifact_access: decoded.artifact_access.clone(),
                    workload: *workload,
                    mesh: completed.result.artifact.clone(),
                    prestress_state: None,
                    chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
                    inventory_limits: ObjectInventoryLimits::default(),
                    required_capabilities: Default::default(),
                    accelerator_requests: analysis_accelerator_requests.clone(),
                },
                policy,
            )
            .await
            {
                Ok(completion) => completion,
                Err(error) => return Ok(sequence_error_composition(&pool, error.to_string())),
            };
            crate::exact_structural_study_root::package_exact_structural_study_result(
                request,
                &decoded,
                &completed.result,
                &boundary_objects,
                crate::exact_structural_study_root::ExactStructuralAnalysisPublication::StructuralModal {
                    signals: &completed_signals,
                    completion: &completion,
                    unresolved_workloads_digest,
                },
            )?
        }
        StructuralStudyWorkloads::LinearSequence { workloads } => {
            if matches!(
                &recovery,
                StructuralTransientRecoveryAdmission::Available {
                    restored: Some(_),
                    ..
                }
            ) {
                return Ok(RemoteExactStructuralStudyComposition::Failed(
                    "linear structural study cannot consume a transient recovery closure".into(),
                ));
            }
            let sequence = match crate::execute_structural_sequence_remotely(
                Arc::clone(&pool),
                StructuralSequencePreparation {
                    authored_study_digest: decoded.authored_study_digest,
                    frozen_project_digest: decoded.frozen_project_digest,
                    program_revision: request.recipe.program_revision.clone(),
                    artifact_access: decoded.artifact_access.clone(),
                    workloads,
                    mesh: Some(completed.result.artifact.clone()),
                    chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
                    inventory_limits: ObjectInventoryLimits::default(),
                    required_capabilities: Default::default(),
                    accelerator_requests: analysis_accelerator_requests.clone(),
                },
                policy,
            )
            .await
            {
                Ok(sequence) => sequence,
                Err(error) => return Ok(sequence_error_composition(&pool, error.to_string())),
            };
            crate::exact_structural_study_root::package_exact_structural_study_result(
                request,
                &decoded,
                &completed.result,
                &boundary_objects,
                crate::exact_structural_study_root::ExactStructuralAnalysisPublication::LinearSequence {
                    signals: &completed_signals,
                    sequence: &sequence,
                    unresolved_workloads_digest,
                },
            )?
        }
        StructuralStudyWorkloads::StructuralTransient {
            workload,
            block_plan,
        } => {
            let (recovered, recovery_publisher) = match recovery {
                StructuralTransientRecoveryAdmission::Available {
                    restored,
                    publisher,
                } => (restored, publisher),
                StructuralTransientRecoveryAdmission::Unavailable => {
                    return Ok(RemoteExactStructuralStudyComposition::Failed(
                        "structural transient execution requires a transactional recovery publisher"
                            .into(),
                    ))
                }
            };
            let recovery = recovered
                .map(|restored| {
                    StructuralTransientRecoveryClosure::from_objects(
                        &decoded.artifact_access,
                        restored.root.clone(),
                        restored.objects.clone(),
                    )
                    .map_err(protocol)
                })
                .transpose()?;
            let preparation = runmat_analysis_execution::prepare_structural_transient_sequence(
                StructuralTransientSequencePreparationInput {
                    workload: &workload,
                    mesh: &completed.result.artifact,
                    modal_state: None,
                    block_plan,
                    initial_dependencies: Vec::new(),
                    artifact_access: decoded.artifact_access.clone(),
                    program_revision: request.recipe.program_revision.clone(),
                    chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
                    inventory_limits: ObjectInventoryLimits::default(),
                },
            )
            .map_err(protocol)?;
            let completion = match crate::execute_structural_transient_remotely(
                Arc::clone(&pool),
                preparation,
                policy,
                recovery.as_ref(),
                recovery_publisher,
            )
            .await
            {
                Ok(completion) => completion,
                Err(error) => return Ok(sequence_error_composition(&pool, error.to_string())),
            };
            crate::exact_structural_study_root::package_exact_structural_study_result(
                request,
                &decoded,
                &completed.result,
                &boundary_objects,
                crate::exact_structural_study_root::ExactStructuralAnalysisPublication::StructuralTransient {
                    signals: &completed_signals,
                    completion: &completion,
                    unresolved_workloads_digest,
                },
            )?
        }
        StructuralStudyWorkloads::StructuralFrequency { workload } => {
            if matches!(
                &recovery,
                StructuralTransientRecoveryAdmission::Available {
                    restored: Some(_),
                    ..
                }
            ) {
                return Ok(RemoteExactStructuralStudyComposition::Failed(
                    "structural frequency study cannot consume a transient recovery closure".into(),
                ));
            }
            if !matches!(
                workload.solver(),
                StructuralFrequencySolverPlan::Direct { .. }
            ) {
                return Ok(RemoteExactStructuralStudyComposition::Failed(
                    "standalone structural frequency execution requires a direct solver plan"
                        .into(),
                ));
            }
            let completion = match crate::execute_structural_frequency_remotely(
                Arc::clone(&pool),
                crate::StructuralFrequencyExecutionPreparation {
                    workload: *workload,
                    mesh: completed.result.artifact.clone(),
                    prestress_state: None,
                    modal_state: None,
                    program_revision: request.recipe.program_revision.clone(),
                    artifact_access: decoded.artifact_access.clone(),
                    chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
                    inventory_limits: ObjectInventoryLimits::default(),
                    observation_limits:
                        runmat_analysis_fea::results::observation::ObservationLimits::default(),
                },
                policy,
            )
            .await
            {
                Ok(completion) => completion,
                Err(error) => return Ok(sequence_error_composition(&pool, error.to_string())),
            };
            crate::exact_structural_study_root::package_exact_structural_study_result(
                request,
                &decoded,
                &completed.result,
                &boundary_objects,
                crate::exact_structural_study_root::ExactStructuralAnalysisPublication::StructuralFrequency {
                    signals: &completed_signals,
                    completion: &completion,
                    unresolved_workloads_digest,
                },
            )?
        }
    };
    for object in objects {
        pool.register_execution_object(object.reference, object.bytes)?;
    }
    Ok(RemoteExactStructuralStudyComposition::Completed(response))
}

fn admit_analysis_task_resources(
    resources: &AllocationResources,
    expected_accelerators: &[runmat_execution::resource::AcceleratorRequest],
) -> Result<runmat_execution::resource::ResourceRequest, String> {
    let task_resources =
        super::pool_resources::task_resources(resources).map_err(|error| error.to_string())?;
    if task_resources.accelerators != expected_accelerators {
        return Err(
            "allocated accelerators differ from the immutable structural-study request".into(),
        );
    }
    Ok(task_resources)
}

#[cfg(test)]
mod allocation_tests {
    use super::*;

    fn resources(accelerator_count: u32) -> AllocationResources {
        AllocationResources {
            cpu_millicores: 500,
            memory_bytes: 1 << 20,
            scratch_bytes: 1 << 20,
            accelerator_count,
            accelerator_class: (accelerator_count > 0).then(|| "gpu.test".into()),
            accelerator_memory_bytes: if accelerator_count > 0 { 1 << 18 } else { 0 },
            maximum_wall_millis: 10_000,
        }
    }

    #[test]
    fn remote_allocation_must_match_the_immutable_study_request() {
        assert!(admit_analysis_task_resources(&resources(0), &[]).is_ok());
        assert!(admit_analysis_task_resources(&resources(1), &[])
            .unwrap_err()
            .contains("immutable structural-study request"));
        let requested = runmat_execution::resource::AcceleratorRequest {
            class: "gpu.test".into(),
            count: 1,
            memory_bytes_each: 1 << 18,
        };
        assert!(
            admit_analysis_task_resources(&resources(0), std::slice::from_ref(&requested))
                .unwrap_err()
                .contains("immutable structural-study request")
        );
        assert_eq!(
            admit_analysis_task_resources(&resources(1), &[requested])
                .unwrap()
                .accelerators
                .len(),
            1
        );
    }
}

fn sequence_error_composition(
    pool: &RemotePoolDriver,
    message: String,
) -> RemoteExactStructuralStudyComposition {
    let snapshot = pool.snapshot();
    if snapshot
        .tasks
        .values()
        .any(|task| task.state == TaskState::Indeterminate)
    {
        RemoteExactStructuralStudyComposition::Indeterminate(message)
    } else if snapshot
        .tasks
        .values()
        .any(|task| task.state == TaskState::Cancelled)
    {
        RemoteExactStructuralStudyComposition::Cancelled
    } else {
        RemoteExactStructuralStudyComposition::Failed(message)
    }
}

fn signal_composition(
    outcome: super::signal_materialization::RemoteSignalMaterializationOutcome,
) -> RemoteExactStructuralStudyComposition {
    match outcome {
        super::signal_materialization::RemoteSignalMaterializationOutcome::Completed(_) => {
            RemoteExactStructuralStudyComposition::Failed(
                "completed signal materialization was used as a terminal outcome".into(),
            )
        }
        super::signal_materialization::RemoteSignalMaterializationOutcome::Failed(message) => {
            RemoteExactStructuralStudyComposition::Failed(message)
        }
        super::signal_materialization::RemoteSignalMaterializationOutcome::Cancelled => {
            RemoteExactStructuralStudyComposition::Cancelled
        }
        super::signal_materialization::RemoteSignalMaterializationOutcome::Indeterminate(
            message,
        ) => RemoteExactStructuralStudyComposition::Indeterminate(message),
    }
}

pub(super) fn composition(
    outcome: RemotePoolExecutionOutcome,
) -> RemoteExactStructuralStudyComposition {
    match outcome {
        RemotePoolExecutionOutcome::Completed(result) => match result.response {
            ProgramExecutionResponse::Failure { message } => {
                RemoteExactStructuralStudyComposition::Failed(message)
            }
            response => RemoteExactStructuralStudyComposition::Completed(response),
        },
        RemotePoolExecutionOutcome::Cancelled => RemoteExactStructuralStudyComposition::Cancelled,
        RemotePoolExecutionOutcome::Indeterminate(message) => {
            RemoteExactStructuralStudyComposition::Indeterminate(message)
        }
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

fn resolve_model_boundary_aliases(
    model: &mut runmat_analysis_core::AnalysisModel,
    resolution: &BoundaryRegionResolution,
) -> Result<(), String> {
    let aliases = runmat_analysis_core::BoundaryRegionAliases::new(resolution.aliases.clone())
        .map_err(|error| error.to_string())?;
    *model = runmat_analysis_core::resolve_boundary_region_aliases(model, &aliases)
        .map_err(|error| error.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use runmat_analysis_fea::execution::StructuralLinearWorkload;
    use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
    use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
    use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
    use runmat_analysis_fea::ComputeBackend;
    use runmat_execution::Digest;
    use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};

    use super::*;

    #[test]
    fn unresolved_importer_alias_is_a_coordinator_resolution_failure() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let step_id = model.sequence.steps[0].step_id.clone();
        let mut workload = StructuralLinearWorkload {
            schema_version: 1,
            model,
            step_id,
            backend: ComputeBackend::Cpu,
            formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
            preconditioner: SpdPreconditionerKind::Jacobi,
            algebra_backend: LinearAlgebraBackendKind::CpuReference,
            observation_execution:
                runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
            require_solver_mesh_for_solid: true,
            initial_guess_transition: None,
        };
        workload.model.constraints[0].visit_targets_mut(|target| {
            *target = runmat_analysis_core::RegionTarget::boundary("face_000002")
        });
        let resolution = BoundaryRegionResolution {
            schema_version: 1,
            geometry_source_digest: Digest::sha256(b"geometry"),
            solver_mesh_digest: Digest::sha256(b"mesh"),
            aliases: BTreeMap::from([(
                "face_000001".into(),
                PersistentEntityId {
                    kind: PersistentEntityKind::Face,
                    source_topology_id: "face-a".into(),
                    assembly_path: vec!["root".into()],
                },
            )]),
        };

        let error = resolve_model_boundary_aliases(&mut workload.model, &resolution).unwrap_err();
        assert!(error.contains("face_000002"));
    }
}
