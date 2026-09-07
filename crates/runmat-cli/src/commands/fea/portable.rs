use std::collections::BTreeSet;

use anyhow::Result;
use futures::stream::{FuturesUnordered, StreamExt};
use runmat_analysis_execution::StructuralSequencePreparation;
use runmat_analysis_fea::execution::DEFAULT_ANALYSIS_CHUNK_BYTES;
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_execution::resource::{Capability, ResourceRequest};
use runmat_execution::task::RetryPolicy;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ArtifactAccess;
use runmat_execution_runner_native::{
    execute_structural_sequence_locally, NativeExecutionConfig, StructuralSequenceExecutionPolicy,
};
use runmat_runtime::analysis::{
    analysis_study_content_digest, AnalysisStudySpec, PreparedStructuralStudy,
    StructuralStudyExecutionFailure, StructuralStudyPlacementEntry,
    StructuralStudyPlacementExecutor, StructuralStudyPlacementFuture,
    StructuralStudyPlacementResult, StructuralStudySequenceNumericalResult,
    StructuralStudyStepNumericalResult,
};

pub(super) struct LocalStructuralStudyPlacement<'a> {
    config: &'a RunMatRuntimeConfig,
}

impl<'a> LocalStructuralStudyPlacement<'a> {
    pub(super) fn new(config: &'a RunMatRuntimeConfig) -> Self {
        Self { config }
    }
}

impl StructuralStudyPlacementExecutor for LocalStructuralStudyPlacement<'_> {
    fn execute<'a>(
        &'a self,
        entries: &'a [StructuralStudyPlacementEntry<'a>],
    ) -> StructuralStudyPlacementFuture<'a> {
        Box::pin(async move {
            if entries.is_empty() {
                return Err("local structural placement requires at least one study".into());
            }
            let available_workers = NativeExecutionConfig::for_current_executable()
                .map_err(|error| error.to_string())?
                .max_workers
                .max(1);
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
                    pending.push(execute_entry(entry, self.config, workers_per_study));
                }
                results.extend(pending.collect::<Vec<_>>().await);
            }
            Ok(results)
        })
    }
}

async fn execute_entry(
    entry: &StructuralStudyPlacementEntry<'_>,
    config: &RunMatRuntimeConfig,
    workers_per_study: u32,
) -> StructuralStudyPlacementResult {
    let outcome = match execute_one(entry.study, entry.prepared, config, workers_per_study).await {
        Ok(result) => Ok(result),
        Err(error) => Err(StructuralStudyExecutionFailure::Failed(error.to_string())),
    };
    StructuralStudyPlacementResult {
        study_index: entry.study_index,
        outcome,
    }
}

async fn execute_one(
    study: &AnalysisStudySpec,
    prepared: &PreparedStructuralStudy,
    config: &RunMatRuntimeConfig,
    workers_per_study: u32,
) -> Result<StructuralStudySequenceNumericalResult> {
    let study_digest = analysis_study_content_digest(study);
    let revision = ProgramRevision::new(
        study_digest,
        study_digest,
        runmat_core::program_environment(crate::diagnostics::parser_compat(config.language.compat)),
    )?;
    let workloads = prepared.workloads();
    let maximum_wall_millis = workloads
        .values()
        .map(|workload| {
            workload
                .procedure_step()
                .map(|step| step.accuracy.resources.maximum_wall_time_milliseconds)
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or_else(|| anyhow::anyhow!("prepared structural study has no workloads"))?;
    let mut required_capabilities = BTreeSet::new();
    required_capabilities.insert(Capability::ProcessIsolation);
    let mut execution_config = NativeExecutionConfig::for_current_executable()?;
    execution_config.max_workers = workers_per_study.max(1);
    let completed = execute_structural_sequence_locally(
        execution_config,
        StructuralSequencePreparation {
            authored_study_digest: study_digest,
            frozen_project_digest: None,
            program_revision: revision,
            artifact_access: artifact_access(study_digest),
            workloads,
            mesh: prepared.solver_mesh().clone(),
            chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: BTreeSet::new(),
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
                accelerators: Vec::new(),
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
