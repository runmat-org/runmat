use std::collections::BTreeSet;

use futures::stream::{FuturesUnordered, StreamExt};
use runmat_analysis_execution::StructuralSequencePreparation;
use runmat_analysis_fea::execution::{StructuralStudyWorkloads, DEFAULT_ANALYSIS_CHUNK_BYTES};
use runmat_execution::resource::{AcceleratorRequest, Capability, ResourceRequest};
use runmat_execution::task::RetryPolicy;
use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ArtifactPublishedObjectSource};
use runmat_runtime::analysis::{
    analysis_study_content_digest, AnalysisStudySpec, PreparedStructuralStudy,
    StructuralStudyBucklingNumericalResult, StructuralStudyExecutionFailure,
    StructuralStudyFrequencyNumericalResult, StructuralStudyModalNumericalResult,
    StructuralStudyModalTransientNumericalResult, StructuralStudyNumericalResult,
    StructuralStudyPlacementEntry, StructuralStudyPlacementExecutor,
    StructuralStudyPlacementFuture, StructuralStudyPlacementResult,
    StructuralStudyPrestressedFrequencyNumericalResult,
    StructuralStudyPrestressedModalNumericalResult, StructuralStudySequenceNumericalResult,
    StructuralStudyStepNumericalResult,
};

use crate::{
    execute_structural_buckling_study_locally, execute_structural_frequency_locally_with_source,
    execute_structural_modal_locally, execute_structural_modal_transient_locally,
    execute_structural_prestressed_frequency_study_locally_with_source,
    execute_structural_prestressed_modal_study_locally, execute_structural_sequence_locally,
    NativeExecutionConfig, RetainedStructuralTransientRecovery,
    StructuralFrequencyExecutionPreparation, StructuralSequenceExecutionPolicy,
};

/// Native-process placement for runtime-prepared structural studies.
///
/// The runtime retains semantic preparation, sweep policy, numerical admission, and publication.
/// This adapter owns only bounded process placement and translation to the generic task substrate.
pub struct NativeStructuralStudyPlacement {
    config: NativeExecutionConfig,
    program_environment: ProgramEnvironment,
}

impl NativeStructuralStudyPlacement {
    pub fn new(config: NativeExecutionConfig, program_environment: ProgramEnvironment) -> Self {
        Self {
            config,
            program_environment,
        }
    }

    pub fn for_current_executable(
        compatibility_mode: runmat_core::CompatMode,
    ) -> std::io::Result<Self> {
        Ok(Self::new(
            NativeExecutionConfig::for_current_executable()?,
            runmat_core::program_environment(compatibility_mode),
        ))
    }
}

impl StructuralStudyPlacementExecutor for NativeStructuralStudyPlacement {
    fn execute<'a>(
        &'a self,
        entries: &'a [StructuralStudyPlacementEntry<'a>],
    ) -> StructuralStudyPlacementFuture<'a> {
        Box::pin(async move {
            if entries.is_empty() {
                return Err("native structural placement requires at least one study".into());
            }
            let available_workers = self.config.max_workers.max(1);
            let workers_per_study = if entries.len() == 1 {
                available_workers
            } else {
                1
            };
            let concurrent_studies = usize::try_from(available_workers)
                .unwrap_or(usize::MAX)
                .min(entries.len());
            let mut results = Vec::with_capacity(entries.len());
            for batch in entries.chunks(concurrent_studies) {
                let pending = FuturesUnordered::new();
                for entry in batch {
                    pending.push(execute_entry(entry, self, workers_per_study));
                }
                results.extend(pending.collect::<Vec<_>>().await);
            }
            Ok(results)
        })
    }
}

async fn execute_entry(
    entry: &StructuralStudyPlacementEntry<'_>,
    placement: &NativeStructuralStudyPlacement,
    workers_per_study: u32,
) -> StructuralStudyPlacementResult {
    let outcome = execute_one(entry.study, entry.prepared, placement, workers_per_study)
        .await
        .map_err(|error| match error {
            crate::NativeExecutionError::Cancelled => StructuralStudyExecutionFailure::Cancelled,
            error => StructuralStudyExecutionFailure::Failed(error.to_string()),
        });
    StructuralStudyPlacementResult {
        study_index: entry.study_index,
        outcome,
    }
}

async fn execute_one(
    study: &AnalysisStudySpec,
    prepared: &PreparedStructuralStudy,
    placement: &NativeStructuralStudyPlacement,
    workers_per_study: u32,
) -> crate::NativeExecutionResult<StructuralStudyNumericalResult> {
    let study_digest = analysis_study_content_digest(study);
    let revision = ProgramRevision::new(
        study_digest,
        study_digest,
        placement.program_environment.clone(),
    )
    .map_err(|error| crate::NativeExecutionError::Configuration(error.to_string()))?;
    let workloads = prepared.workloads();
    let accelerator_requests = local_accelerator_requests(&workloads)?;
    let maximum_wall_millis = workloads
        .authored_model()
        .map_err(|error| crate::NativeExecutionError::Configuration(error.to_string()))?
        .sequence
        .steps
        .iter()
        .map(|step| step.accuracy.resources.maximum_wall_time_milliseconds)
        .max()
        .ok_or_else(|| {
            crate::NativeExecutionError::Configuration(
                "prepared structural study has no workloads".into(),
            )
        })?;
    let mut required_capabilities = BTreeSet::new();
    required_capabilities.insert(Capability::ProcessIsolation);
    let mut execution_config = placement.config.clone();
    execution_config.max_workers = workers_per_study.max(1);
    let policy = StructuralSequenceExecutionPolicy {
        resources: ResourceRequest {
            cpu_millicores: 1_000,
            memory_bytes: 1024 * 1024 * 1024,
            scratch_bytes: 1024 * 1024 * 1024,
            max_wall_millis: maximum_wall_millis,
            max_artifact_bytes: ObjectInventoryLimits::default().max_total_bytes,
            max_egress_bytes: 0,
            max_relay_bytes: 0,
            accelerators: accelerator_requests.clone(),
            required_capabilities,
        },
        retry: RetryPolicy::IdempotentInfrastructure,
        deadline_unix_millis: None,
        priority: 0,
    };
    let access = artifact_access(study_digest);
    match workloads {
        StructuralStudyWorkloads::HeterogeneousSequence { workloads } => {
            let authored = StructuralStudyWorkloads::HeterogeneousSequence { workloads };
            let step_ids = authored.step_ids().map_err(|error| {
                crate::NativeExecutionError::Configuration(error.to_string())
            })?;
            let [producer_step_id, consumer_step_id] = step_ids.as_slice() else {
                return Err(crate::NativeExecutionError::Configuration(
                    "heterogeneous structural study requires exactly two authored steps".into(),
                ));
            };
            let producer = match &authored {
                StructuralStudyWorkloads::HeterogeneousSequence { workloads } => workloads
                    .get(producer_step_id)
                    .ok_or_else(|| crate::NativeExecutionError::Configuration(
                        "heterogeneous structural producer is absent".into(),
                    ))?,
                _ => unreachable!(),
            };
            let consumer_is_prestressed_modal = match &authored {
                StructuralStudyWorkloads::HeterogeneousSequence { workloads } => matches!(
                    workloads.get(consumer_step_id),
                    Some(
                        runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedModal {
                            ..
                        }
                    )
                ),
                _ => false,
            };
            let consumer_is_prestressed_frequency = match &authored {
                StructuralStudyWorkloads::HeterogeneousSequence { workloads } => matches!(
                    workloads.get(consumer_step_id),
                    Some(
                        runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedFrequency {
                            ..
                        }
                    )
                ),
                _ => false,
            };
            if matches!(
                producer,
                runmat_analysis_fea::execution::StructuralStepWorkload::Linear { .. }
            ) {
                if consumer_is_prestressed_frequency {
                    let (completed, source) =
                        execute_structural_prestressed_frequency_study_locally_with_source(
                            execution_config,
                            runmat_analysis_execution::StructuralPrestressedFrequencyStudyPreparation {
                                authored_study_digest: study_digest,
                                frozen_project_digest: None,
                                workloads: authored,
                                mesh: prepared.solver_mesh().clone(),
                                artifact_access: access.clone(),
                                program_revision: revision,
                                chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
                                inventory_limits: ObjectInventoryLimits::default(),
                                required_capabilities: BTreeSet::new(),
                                accelerator_requests,
                            },
                            policy,
                        )
                        .await?;
                    let (catalog, artifact_roots) =
                        completed.frequency.published_artifacts(&access)?;
                    let frequency = StructuralStudyFrequencyNumericalResult::new(
                        None,
                        completed.resolved_workload,
                        completed.frequency.result.result,
                        completed.frequency.publication,
                        catalog,
                        artifact_roots,
                        ArtifactPublishedObjectSource::new(source),
                    )
                    .map_err(crate::NativeExecutionError::Configuration)?;
                    return Ok(
                        StructuralStudyNumericalResult::StructuralPrestressedFrequency(Box::new(
                            StructuralStudyPrestressedFrequencyNumericalResult {
                                frozen_project_digest: None,
                                preload_step_id: producer_step_id.clone(),
                                frequency_step_id: consumer_step_id.clone(),
                                preload_result: completed.preload.result.run,
                                preload_solver_result_identity: completed.preload.identity,
                                prestress_state: completed.prestress_state.state,
                                frequency,
                            },
                        )),
                    );
                }
                if consumer_is_prestressed_modal {
                    let completed = execute_structural_prestressed_modal_study_locally(
                        execution_config,
                        runmat_analysis_execution::StructuralPrestressedModalStudyPreparation {
                            authored_study_digest: study_digest,
                            frozen_project_digest: None,
                            workloads: authored,
                            mesh: prepared.solver_mesh().clone(),
                            artifact_access: access,
                            program_revision: revision,
                            chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
                            inventory_limits: ObjectInventoryLimits::default(),
                            required_capabilities: BTreeSet::new(),
                            accelerator_requests,
                        },
                        policy,
                    )
                    .await?;
                    return Ok(StructuralStudyNumericalResult::StructuralPrestressedModal(
                        Box::new(StructuralStudyPrestressedModalNumericalResult {
                            frozen_project_digest: None,
                            preload_step_id: producer_step_id.clone(),
                            modal_step_id: consumer_step_id.clone(),
                            preload_result: completed.preload.result.run,
                            preload_solver_result_identity: completed.preload.identity,
                            prestress_state: completed.prestress_state.state,
                            modal_result: completed.modal.result,
                            modal_numerical_receipt: completed.modal.numerical_receipt,
                        }),
                    ));
                }
                let completed = execute_structural_buckling_study_locally(
                    execution_config,
                    runmat_analysis_execution::StructuralBucklingStudyPreparation {
                        authored_study_digest: study_digest,
                        frozen_project_digest: None,
                        workloads: authored,
                        mesh: prepared.solver_mesh().clone(),
                        artifact_access: access,
                        program_revision: revision,
                        chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
                        inventory_limits: ObjectInventoryLimits::default(),
                        required_capabilities: BTreeSet::new(),
                        accelerator_requests,
                    },
                    policy,
                )
                .await?;
                return Ok(StructuralStudyNumericalResult::StructuralBuckling(Box::new(
                    StructuralStudyBucklingNumericalResult {
                        frozen_project_digest: None,
                        preload_step_id: producer_step_id.clone(),
                        buckling_step_id: consumer_step_id.clone(),
                        preload_result: completed.preload.result.run,
                        preload_solver_result_identity: completed.preload.identity,
                        prestress_state: completed.prestress_state.state,
                        buckling_result: completed.buckling.result,
                        buckling_numerical_receipt: completed.buckling.numerical_receipt,
                    },
                )));
            }
            let mut publisher = RetainedStructuralTransientRecovery::default();
            let completed = execute_structural_modal_transient_locally(
                execution_config,
                runmat_analysis_execution::StructuralModalTransientStudyPreparation {
                    authored_study_digest: study_digest,
                    frozen_project_digest: None,
                    workloads: authored,
                    mesh: prepared.solver_mesh().clone(),
                    artifact_access: access,
                    program_revision: revision,
                    chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
                    inventory_limits: ObjectInventoryLimits::default(),
                    required_capabilities: BTreeSet::new(),
                    accelerator_requests,
                },
                policy,
                None,
                &mut publisher,
            )
            .await?;
            Ok(StructuralStudyNumericalResult::StructuralModalTransient(
                Box::new(StructuralStudyModalTransientNumericalResult {
                    frozen_project_digest: None,
                    modal_step_id: producer_step_id.clone(),
                    transient_step_id: consumer_step_id.clone(),
                    modal_result: completed.modal.result,
                    modal_numerical_receipt: completed.modal.numerical_receipt,
                    basis_state: completed.basis_state.state,
                    transient_result: completed.transient.result.result,
                    transient_numerical_receipt: completed
                        .transient
                        .result
                        .numerical_receipt,
                }),
            ))
        }
        StructuralStudyWorkloads::LinearSequence { workloads } => {
            let completed = execute_structural_sequence_locally(
                execution_config,
                StructuralSequencePreparation {
                    authored_study_digest: study_digest,
                    frozen_project_digest: None,
                    program_revision: revision,
                    artifact_access: access,
                    workloads,
                    mesh: Some(prepared.solver_mesh().clone()),
                    chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
                    inventory_limits: ObjectInventoryLimits::default(),
                    required_capabilities: BTreeSet::new(),
                    accelerator_requests,
                },
                policy,
            )
            .await?;
            Ok(StructuralStudyNumericalResult::LinearSequence(
                StructuralStudySequenceNumericalResult {
                    frozen_project_digest: None,
                    steps: completed
                        .steps
                        .into_iter()
                        .map(|step| StructuralStudyStepNumericalResult {
                            step_id: step.step_id,
                            solver_result_identity: step.result_identity,
                            result: step.result.run,
                        })
                        .collect(),
                    states: completed
                        .states
                        .into_iter()
                        .map(|state| state.state)
                        .collect(),
                },
            ))
        }
        StructuralStudyWorkloads::StructuralFrequency { workload } => {
            let resolved_workload = *workload;
            let (completion, source) = execute_structural_frequency_locally_with_source(
                execution_config,
                StructuralFrequencyExecutionPreparation {
                    workload: resolved_workload.clone(),
                    mesh: prepared.solver_mesh().clone(),
                    prestress_state: None,
                    modal_state: None,
                    program_revision: revision,
                    artifact_access: access.clone(),
                    chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
                    inventory_limits: ObjectInventoryLimits::default(),
                    observation_limits:
                        runmat_analysis_fea::results::observation::ObservationLimits::default(),
                },
                policy,
            )
            .await?;
            let (catalog, artifact_roots) = completion.published_artifacts(&access)?;
            Ok(StructuralStudyNumericalResult::StructuralFrequency(
                Box::new(
                    StructuralStudyFrequencyNumericalResult::new(
                        None,
                        resolved_workload,
                        completion.result.result,
                        completion.publication,
                        catalog,
                        artifact_roots,
                        ArtifactPublishedObjectSource::new(source),
                    )
                    .map_err(crate::NativeExecutionError::Configuration)?,
                ),
            ))
        }
        StructuralStudyWorkloads::StructuralModal { workload } => {
            let step_id = workload.step_id().clone();
            let completed = execute_structural_modal_locally(
                execution_config,
                runmat_analysis_execution::StructuralModalTaskPreparation {
                    authored_study_digest: study_digest,
                    frozen_project_digest: None,
                    program_revision: revision,
                    artifact_access: access,
                    workload: *workload,
                    mesh: prepared.solver_mesh().clone(),
                    prestress_state: None,
                    chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
                    inventory_limits: ObjectInventoryLimits::default(),
                    required_capabilities: BTreeSet::new(),
                    accelerator_requests,
                },
                policy,
            )
            .await?;
            Ok(StructuralStudyNumericalResult::StructuralModal(Box::new(
                StructuralStudyModalNumericalResult {
                    frozen_project_digest: None,
                    step_id,
                    result: completed.result.result,
                    numerical_receipt: completed.result.numerical_receipt,
                },
            )))
        }
        StructuralStudyWorkloads::StructuralTransient { .. } => {
            Err(crate::NativeExecutionError::Configuration(
                "local placement does not execute structural transient studies through this boundary"
                    .into(),
            ))
        }
    }
}

fn local_accelerator_requests(
    workloads: &StructuralStudyWorkloads,
) -> crate::NativeExecutionResult<Vec<AcceleratorRequest>> {
    if workloads
        .backends()
        .map_err(|error| crate::NativeExecutionError::Configuration(error.to_string()))?
        .into_iter()
        .all(|backend| backend != runmat_analysis_fea::ComputeBackend::Gpu)
    {
        return Ok(Vec::new());
    }
    Ok(vec![
        crate::accelerator_placement::active_provider_resource()?,
    ])
}

fn artifact_access(study_digest: Digest) -> ArtifactAccess {
    ArtifactAccess {
        authorization_scope: format!("study:{study_digest}"),
        encryption_context: Digest::sha256(
            [
                b"runmat-local-analysis-access/v1\0".as_slice(),
                study_digest.bytes(),
            ]
            .concat(),
        ),
    }
}
