//! Local-process adapter for the meshing-owned exact DAG.

mod cache;
mod completion;

use std::collections::BTreeSet;
use std::time::Duration;

use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::{ArtifactResult, LogicalObject};
use runmat_execution_runner::AttemptSuccess;
use runmat_meshing_core::{CanonicalMeshingContract, MeshingProgress, StableDigest};
use runmat_meshing_execution::{
    build_task_submission, ExactMeshingDagExecutor, MeshingExecutionContext, MeshingHostResponse,
    MeshingStageExecutionError, MeshingTaskEffectPolicy, PlannedMeshingStage,
};

use crate::{NativeProgramSession, NativeProgramTask, ProgramProgress};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeshingExecutionPolicy {
    pub cpu_millicores: u32,
    pub reuse_accepted_stages: bool,
    pub maximum_egress_bytes: u64,
    pub maximum_relay_bytes: u64,
    pub deadline_unix_millis: Option<u64>,
    pub priority: i16,
    pub maximum_attempts: u16,
    pub poll_interval: Duration,
}

impl Default for MeshingExecutionPolicy {
    fn default() -> Self {
        Self {
            cpu_millicores: crate::machine_capacity::cpu_millicores(),
            reuse_accepted_stages: true,
            maximum_egress_bytes: 0,
            maximum_relay_bytes: 0,
            deadline_unix_millis: None,
            priority: 0,
            maximum_attempts: 2,
            poll_interval: Duration::from_millis(5),
        }
    }
}

pub struct NativeExactMeshingExecutor<'a> {
    session: &'a NativeProgramSession,
    policy: MeshingExecutionPolicy,
    progress: Vec<ProgramProgress>,
    cache: Option<cache::NativeMeshingStageCache>,
}

impl<'a> NativeExactMeshingExecutor<'a> {
    pub fn new(
        session: &'a NativeProgramSession,
        policy: MeshingExecutionPolicy,
    ) -> Result<Self, MeshingStageExecutionError> {
        policy.validate()?;
        Ok(Self {
            session,
            policy,
            progress: Vec::new(),
            cache: if policy.reuse_accepted_stages {
                cache::NativeMeshingStageCache::open(&session.meshing_cache_root)?
            } else {
                None
            },
        })
    }

    pub fn drain_progress(&mut self) -> Vec<ProgramProgress> {
        std::mem::take(&mut self.progress)
    }
}

impl MeshingExecutionPolicy {
    pub(crate) fn validate(self) -> Result<(), MeshingStageExecutionError> {
        if self.cpu_millicores == 0
            || self.maximum_attempts < 2
            || self.maximum_attempts > 16
            || self.poll_interval.is_zero()
        {
            return Err(MeshingStageExecutionError::new(
                "meshing execution policy contains an invalid bound",
            ));
        }
        Ok(())
    }
}

impl NativeExactMeshingExecutor<'_> {
    fn submit_stage(
        &self,
        stage: &PlannedMeshingStage,
        revision: ProgramRevision,
    ) -> Result<NativeProgramTask, MeshingStageExecutionError> {
        let invocation = stage.prepare_invocation(revision).map_err(stage_error)?;
        let mut object_store = self.session.object_store();
        invocation
            .write_to(&mut object_store)
            .map_err(stage_error)?;
        let program = invocation.into_request();
        let effect =
            if stage.host().workload.stage == runmat_meshing_core::MeshingStageKind::Publication {
                MeshingTaskEffectPolicy::UnknownEffect
            } else {
                MeshingTaskEffectPolicy::ContentAddressedPure {
                    maximum_attempts: self.policy.maximum_attempts,
                    replay_proof_digest: StableDigest::from_bytes(*program.artifact.id.0.bytes()),
                }
            };
        let submission = build_task_submission(
            stage.host(),
            &program,
            BTreeSet::new(),
            &MeshingExecutionContext {
                scope_id: self.session.scope_id(),
                pool_id: self.session.pool_id(),
                permitted_trust_tiers: BTreeSet::from([
                    runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
                ]),
                artifact_access: stage.host().artifact_access.clone(),
                cpu_millicores: stage_cpu_millicores(self.policy.cpu_millicores, stage.host()),
                maximum_egress_bytes: self.policy.maximum_egress_bytes,
                maximum_relay_bytes: self.policy.maximum_relay_bytes,
                deadline_unix_millis: self.policy.deadline_unix_millis,
                priority: self.policy.priority,
            },
            effect,
        )
        .map_err(stage_error)?;
        self.session
            .submit(program, submission)
            .map_err(stage_error)
    }

    fn wait_for_task(
        &mut self,
        task: NativeProgramTask,
        stage: &PlannedMeshingStage,
    ) -> Result<AttemptSuccess, MeshingStageExecutionError> {
        let workload = &stage.host().workload;
        let stage_span = tracing::info_span!(
            "meshing.stage",
            stage = ?workload.stage,
            partition_kind = ?workload.partition.kind,
            partition_index = workload.partition.partition_index,
            partition_count = workload.partition.partition_count,
        );
        let _entered = stage_span.enter();
        tracing::info!(
            event_code = "meshing.stage.started",
            "meshing stage started"
        );
        loop {
            self.record_progress(task.drain_progress());
            if let Some(completion) = task.try_completion() {
                let result = completion.into_delivery_result();
                self.record_progress(task.drain_progress());
                return match result {
                    Ok(success) => {
                        tracing::info!(
                            event_code = "meshing.stage.completed",
                            outcome = "success",
                            "meshing stage completed"
                        );
                        Ok(success)
                    }
                    Err(error) => {
                        tracing::error!(
                            event_code = "meshing.stage.completed",
                            outcome = "failure",
                            error = %error,
                            "meshing stage failed"
                        );
                        Err(completion::stage_failure(error, workload.stage))
                    }
                };
            }
            std::thread::sleep(self.policy.poll_interval);
        }
    }

    fn record_progress(&mut self, records: Vec<ProgramProgress>) {
        for record in &records {
            if let Ok(progress) = MeshingProgress::canonical_decode(&record.payload) {
                let checkpoint = tracing::info_span!(
                    "meshing.progress",
                    stage = ?progress.stage,
                    partition_index = progress.partition_index,
                    sequence = progress.sequence,
                    completed_work = progress.completed_work,
                    estimated_work = progress.estimated_work,
                    elapsed_time_ms = progress.elapsed_time_ms,
                    consumed_search_work = progress.consumed_search_work,
                    peak_memory_bytes = progress.peak_memory_bytes,
                    entity_counts = ?progress.entity_counts,
                );
                let _entered = checkpoint.enter();
                tracing::info!(
                    event_code = "meshing.stage.progress",
                    "meshing stage progress"
                );
            }
        }
        self.progress.extend(records);
    }
}

impl CacheImport for NativeExactMeshingExecutor<'_> {
    fn read_verified_bounded_with_admission(
        &self,
        digest: runmat_execution::Digest,
        maximum_bytes: u64,
        admit: &mut dyn FnMut(u64) -> runmat_execution_artifact::ArtifactResult<()>,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        self.session
            .object_store()
            .read_verified_bounded_with_admission(digest, maximum_bytes, admit)
    }
    fn read_verified_bounded(
        &self,
        digest: runmat_execution::Digest,
        maximum_bytes: u64,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        self.session
            .object_store()
            .read_verified_bounded(digest, maximum_bytes)
    }
    fn read_verified(&self, digest: runmat_execution::Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.session.object_store().read_verified(digest)
    }
}

impl CacheExport for NativeExactMeshingExecutor<'_> {
    fn write_verified(&mut self, object: &LogicalObject) -> ArtifactResult<()> {
        self.session.object_store().write_verified(object)
    }
}

impl ExactMeshingDagExecutor for NativeExactMeshingExecutor<'_> {
    async fn execute_stage<'a>(
        &'a mut self,
        stage: &'a PlannedMeshingStage,
        revision: ProgramRevision,
    ) -> Result<AttemptSuccess, MeshingStageExecutionError> {
        let started = std::time::Instant::now();
        if let Some(cache) = &self.cache {
            if let Some(success) = cache.restore(
                stage,
                &mut self.session.object_store(),
                self.session,
                self.policy.deadline_unix_millis,
                started,
            )? {
                return Ok(success);
            }
        }
        let task = self.submit_stage(stage, revision)?;
        let success = self.wait_for_task(task, stage)?;
        if let Some(cache) = &mut self.cache {
            cache.retain(
                stage,
                &self.session.object_store(),
                &success,
                self.session,
                self.policy.deadline_unix_millis,
                started,
            )?;
        }
        Ok(success)
    }

    async fn execute_stages<'a>(
        &'a mut self,
        stages: &'a [PlannedMeshingStage],
        revision: ProgramRevision,
    ) -> Result<Vec<AttemptSuccess>, MeshingStageExecutionError> {
        let mut results = Vec::with_capacity(stages.len());
        let mut pending = Vec::new();
        for (index, stage) in stages.iter().enumerate() {
            let started = std::time::Instant::now();
            let hit = match &self.cache {
                Some(cache) => cache.restore(
                    stage,
                    &mut self.session.object_store(),
                    self.session,
                    self.policy.deadline_unix_millis,
                    started,
                )?,
                None => None,
            };
            results.push(hit);
            if results[index].is_none() {
                pending.push((index, self.submit_stage(stage, revision.clone())?, started));
            }
        }
        for (index, task, started) in pending {
            let success = self.wait_for_task(task, &stages[index])?;
            if let Some(cache) = &mut self.cache {
                cache.retain(
                    &stages[index],
                    &self.session.object_store(),
                    &success,
                    self.session,
                    self.policy.deadline_unix_millis,
                    started,
                )?;
            }
            results[index] = Some(success);
        }
        Ok(results
            .into_iter()
            .map(|value| value.expect("every submitted stage completed"))
            .collect())
    }
}

/// CPU reservation for one meshing stage. Only tetrahedralization runs a
/// parallel pool, and each pool thread needs its stack inside the stage's
/// memory reservation, so the thread count never exceeds what memory admits.
pub(crate) fn stage_cpu_millicores(
    policy_cpu_millicores: u32,
    host: &runmat_meshing_execution::MeshingHostWorkload,
) -> u32 {
    if host.workload.stage != runmat_meshing_core::MeshingStageKind::Tetrahedralization {
        return policy_cpu_millicores.min(1_000);
    }
    let memory_threads =
        host.ceilings.maximum_memory_bytes / runmat_cpu_execution::WORKER_STACK_BYTES;
    let memory_millicores =
        u32::try_from(memory_threads.max(1).saturating_mul(1_000)).unwrap_or(u32::MAX);
    policy_cpu_millicores.min(memory_millicores)
}

fn stage_error(error: impl std::fmt::Display) -> MeshingStageExecutionError {
    MeshingStageExecutionError::new(error.to_string())
}
