//! Local-process adapter for the meshing-owned exact DAG.

use std::collections::BTreeSet;
use std::time::Duration;

use runmat_execution::identity::ArtifactId;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::{ArtifactResult, LogicalObject};
use runmat_execution_runner::AttemptSuccess;
use runmat_meshing_core::StableDigest;
use runmat_meshing_execution::{
    build_task_submission, ExactMeshingDagExecutor, MeshingExecutionContext,
    MeshingStageExecutionError, MeshingTaskEffectPolicy, PlannedMeshingStage,
};

use crate::{NativeProgramSession, NativeProgramTask, ProgramProgress};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeMeshingExecutionPolicy {
    pub cpu_millicores: u32,
    pub maximum_egress_bytes: u64,
    pub maximum_relay_bytes: u64,
    pub deadline_unix_millis: Option<u64>,
    pub priority: i16,
    pub maximum_attempts: u16,
    pub poll_interval: Duration,
}

impl Default for NativeMeshingExecutionPolicy {
    fn default() -> Self {
        Self {
            cpu_millicores: 1_000,
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
    policy: NativeMeshingExecutionPolicy,
    progress: Vec<ProgramProgress>,
}

impl<'a> NativeExactMeshingExecutor<'a> {
    pub fn new(
        session: &'a NativeProgramSession,
        policy: NativeMeshingExecutionPolicy,
    ) -> Result<Self, MeshingStageExecutionError> {
        if policy.cpu_millicores == 0
            || policy.maximum_attempts < 2
            || policy.maximum_attempts > 16
            || policy.poll_interval.is_zero()
        {
            return Err(MeshingStageExecutionError::new(
                "native meshing execution policy contains an invalid bound",
            ));
        }
        Ok(Self {
            session,
            policy,
            progress: Vec::new(),
        })
    }

    pub fn drain_progress(&mut self) -> Vec<ProgramProgress> {
        std::mem::take(&mut self.progress)
    }

    fn submit_stage(
        &self,
        stage: &PlannedMeshingStage,
        revision: ProgramRevision,
    ) -> Result<NativeProgramTask, MeshingStageExecutionError> {
        let program = stage.program_request(revision).map_err(stage_error)?;
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
            &stage.host().workload,
            &stage.host().stage_identity,
            &stage.host().resolved_request,
            stage.input_roots(),
            BTreeSet::new(),
            &MeshingExecutionContext {
                scope_id: self.session.scope_id(),
                pool_id: self.session.pool_id(),
                program_artifact_id: ArtifactId::derive(&[program.artifact.id.0.bytes()]),
                artifact_access: stage.host().artifact_access.clone(),
                cpu_millicores: self.policy.cpu_millicores,
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
    ) -> Result<AttemptSuccess, MeshingStageExecutionError> {
        loop {
            self.progress.extend(task.drain_progress());
            if let Some(result) = task.try_result() {
                self.progress.extend(task.drain_progress());
                return result.map_err(MeshingStageExecutionError::new);
            }
            std::thread::sleep(self.policy.poll_interval);
        }
    }
}

impl CacheImport for NativeExactMeshingExecutor<'_> {
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
    fn execute_stage(
        &mut self,
        stage: &PlannedMeshingStage,
        revision: ProgramRevision,
    ) -> Result<AttemptSuccess, MeshingStageExecutionError> {
        let task = self.submit_stage(stage, revision)?;
        self.wait_for_task(task)
    }

    fn execute_stages(
        &mut self,
        stages: &[PlannedMeshingStage],
        revision: ProgramRevision,
    ) -> Result<Vec<AttemptSuccess>, MeshingStageExecutionError> {
        let tasks = stages
            .iter()
            .map(|stage| self.submit_stage(stage, revision.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        tasks
            .into_iter()
            .map(|task| self.wait_for_task(task))
            .collect()
    }
}

fn stage_error(error: impl std::fmt::Display) -> MeshingStageExecutionError {
    MeshingStageExecutionError::new(error.to_string())
}
