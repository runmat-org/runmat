use std::collections::BTreeSet;

use futures::stream::{FuturesUnordered, StreamExt};
use runmat_analysis_execution::StructuralSequencePreparation;
use runmat_analysis_fea::execution::DEFAULT_ANALYSIS_CHUNK_BYTES;
use runmat_execution::resource::{AcceleratorRequest, Capability, ResourceRequest};
use runmat_execution::task::RetryPolicy;
use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ArtifactAccess;
use runmat_runtime::analysis::{
    analysis_study_content_digest, AnalysisStudySpec, PreparedStructuralStudy,
    StructuralStudyExecutionFailure, StructuralStudyPlacementEntry,
    StructuralStudyPlacementExecutor, StructuralStudyPlacementFuture,
    StructuralStudyPlacementResult, StructuralStudySequenceNumericalResult,
    StructuralStudyStepNumericalResult,
};

use crate::{
    execute_structural_sequence_locally, NativeExecutionConfig, StructuralSequenceExecutionPolicy,
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
) -> crate::NativeExecutionResult<StructuralStudySequenceNumericalResult> {
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
        .values()
        .map(|workload| {
            workload
                .procedure_step()
                .map(|step| step.accuracy.resources.maximum_wall_time_milliseconds)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| crate::NativeExecutionError::Configuration(error.to_string()))?
        .into_iter()
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
    let completed = execute_structural_sequence_locally(
        execution_config,
        StructuralSequencePreparation {
            authored_study_digest: study_digest,
            frozen_project_digest: None,
            program_revision: revision,
            artifact_access: artifact_access(study_digest),
            workloads,
            mesh: Some(prepared.solver_mesh().clone()),
            chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: BTreeSet::new(),
            accelerator_requests: accelerator_requests.clone(),
        },
        StructuralSequenceExecutionPolicy {
            resources: ResourceRequest {
                cpu_millicores: 1_000,
                memory_bytes: 1024 * 1024 * 1024,
                scratch_bytes: 1024 * 1024 * 1024,
                max_wall_millis: maximum_wall_millis,
                max_artifact_bytes: ObjectInventoryLimits::default().max_total_bytes,
                max_egress_bytes: 0,
                max_relay_bytes: 0,
                accelerators: accelerator_requests,
                required_capabilities,
            },
            retry: RetryPolicy::IdempotentInfrastructure,
            deadline_unix_millis: None,
            priority: 0,
        },
    )
    .await?;
    Ok(StructuralStudySequenceNumericalResult {
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
    })
}

fn local_accelerator_requests(
    workloads: &std::collections::BTreeMap<
        runmat_analysis_core::ProcedureStepId,
        runmat_analysis_fea::execution::StructuralLinearWorkload,
    >,
) -> crate::NativeExecutionResult<Vec<AcceleratorRequest>> {
    if workloads
        .values()
        .all(|workload| workload.backend != runmat_analysis_fea::ComputeBackend::Gpu)
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
