//! Remote-pool adapter for the meshing-owned exact DAG.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use runmat_execution::value::{ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::cache::{CacheExport, CacheImport, FilesystemObjectStore};
use runmat_execution_artifact::{ArtifactAccess, ArtifactResult, LogicalObject};
use runmat_execution_runner::AttemptSuccess;
use runmat_meshing_core::StableDigest;
use runmat_meshing_execution::{
    build_task_submission, ExactMeshingDagExecutor, MeshingExecutionContext,
    MeshingStageExecutionError, MeshingTaskEffectPolicy, PlannedMeshingStage,
};

use crate::{MeshingExecutionPolicy, ProgramProgress, RemotePoolDriver};

pub struct RemoteExactMeshingExecutor {
    pool: Arc<RemotePoolDriver>,
    artifact_access: ArtifactAccess,
    policy: MeshingExecutionPolicy,
    pending: HashMap<Digest, ValueRef>,
    local_store: FilesystemObjectStore,
    inventory_limits: runmat_execution_artifact::object::ObjectInventoryLimits,
    _directory: tempfile::TempDir,
    progress: Vec<ProgramProgress>,
}

impl RemoteExactMeshingExecutor {
    pub fn new(
        pool: Arc<RemotePoolDriver>,
        artifact_access: ArtifactAccess,
        policy: MeshingExecutionPolicy,
        maximum_object_bytes: u64,
        inventory_limits: runmat_execution_artifact::object::ObjectInventoryLimits,
    ) -> Result<Self, MeshingStageExecutionError> {
        artifact_access.validate().map_err(stage_error)?;
        policy.validate()?;
        if pool.value_scope() != artifact_access.authorization_scope {
            return Err(MeshingStageExecutionError::new(
                "remote pool value scope differs from meshing artifact authority",
            ));
        }
        if maximum_object_bytes == 0
            || inventory_limits.max_object_bytes == 0
            || inventory_limits.max_objects == 0
            || inventory_limits.max_total_bytes == 0
        {
            return Err(MeshingStageExecutionError::new(
                "remote meshing object bound must be nonzero",
            ));
        }
        let directory = tempfile::Builder::new()
            .prefix("runmat-remote-meshing-")
            .tempdir()
            .map_err(stage_error)?;
        let local_store = FilesystemObjectStore::open(
            directory.path().join("objects"),
            maximum_object_bytes.min(inventory_limits.max_object_bytes),
        )
        .map_err(stage_error)?;
        Ok(Self {
            pool,
            artifact_access,
            policy,
            pending: HashMap::new(),
            local_store,
            inventory_limits,
            _directory: directory,
            progress: Vec::new(),
        })
    }

    pub fn drain_progress(&mut self) -> Vec<ProgramProgress> {
        std::mem::take(&mut self.progress)
    }

    fn prepare_stage(
        &mut self,
        stage: &PlannedMeshingStage,
        revision: ProgramRevision,
    ) -> Result<runmat_execution_artifact::ProgramExecutionRequest, MeshingStageExecutionError>
    {
        if stage.host().artifact_access != self.artifact_access {
            return Err(MeshingStageExecutionError::new(
                "planned meshing stage differs from remote executor artifact authority",
            ));
        }
        let invocation = stage.prepare_invocation(revision).map_err(stage_error)?;
        invocation.write_to(self).map_err(stage_error)?;
        let program = invocation.into_request();
        self.register_inputs(&program.arguments)?;
        Ok(program)
    }

    fn register_inputs(
        &mut self,
        arguments: &[ValuePayload],
    ) -> Result<(), MeshingStageExecutionError> {
        for argument in arguments {
            let ValuePayload::Object(reference) = argument else {
                return Err(MeshingStageExecutionError::new(
                    "remote meshing inputs must remain externalized",
                ));
            };
            self.register_reference(reference)?;
            self.pending.remove(&reference.logical_digest);
        }
        let pending = std::mem::take(&mut self.pending);
        for reference in pending.into_values() {
            self.register_reference(&reference)?;
        }
        Ok(())
    }

    fn register_reference(&self, reference: &ValueRef) -> Result<(), MeshingStageExecutionError> {
        if self
            .pool
            .execution_object(reference)
            .map_err(stage_error)?
            .is_some()
        {
            return Ok(());
        }
        let bytes = self
            .read_verified_bounded(reference.logical_digest, reference.encoded_length)
            .map_err(stage_error)?
            .ok_or_else(|| {
                MeshingStageExecutionError::new(format!(
                    "remote meshing input {} is unavailable",
                    reference.logical_digest
                ))
            })?;
        self.pool
            .register_execution_object(reference.clone(), bytes)
            .map_err(stage_error)
    }

    fn submission(
        &self,
        stage: &PlannedMeshingStage,
        program: &runmat_execution_artifact::ProgramExecutionRequest,
    ) -> Result<runmat_execution_runner::TaskSubmission, MeshingStageExecutionError> {
        let effect =
            if stage.host().workload.stage == runmat_meshing_core::MeshingStageKind::Publication {
                MeshingTaskEffectPolicy::UnknownEffect
            } else {
                MeshingTaskEffectPolicy::ContentAddressedPure {
                    maximum_attempts: self.policy.maximum_attempts,
                    replay_proof_digest: StableDigest::from_bytes(*program.artifact.id.0.bytes()),
                }
            };
        build_task_submission(
            stage.host(),
            program,
            BTreeSet::new(),
            &MeshingExecutionContext {
                scope_id: self.pool.scope_id(),
                pool_id: self.pool.pool_id(),
                permitted_trust_tiers: BTreeSet::from([
                    runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
                ]),
                artifact_access: self.artifact_access.clone(),
                cpu_millicores: crate::exact_meshing_executor::stage_cpu_millicores(
                    self.policy.cpu_millicores,
                    stage.host(),
                ),
                maximum_egress_bytes: self.policy.maximum_egress_bytes,
                maximum_relay_bytes: self.policy.maximum_relay_bytes,
                deadline_unix_millis: self.policy.deadline_unix_millis,
                priority: self.policy.priority,
            },
            effect,
        )
        .map_err(stage_error)
    }

    async fn submit_stage(
        &mut self,
        stage: &PlannedMeshingStage,
        revision: ProgramRevision,
        wave: &mut crate::remote::RemoteObservedWave,
    ) -> Result<crate::RemoteTaskCompletion, MeshingStageExecutionError> {
        let program = self.prepare_stage(stage, revision)?;
        let submission = self.submission(stage, &program)?;
        wave.arm(&[submission.request.id]).map_err(stage_error)?;
        self.pool.submit(submission, program).map_err(stage_error)
    }

    async fn finish(
        &mut self,
        completion: crate::RemoteTaskCompletion,
        stage: &PlannedMeshingStage,
    ) -> Result<AttemptSuccess, MeshingStageExecutionError> {
        let reused_invocation = completion.reused_invocation();
        let (result, progress) = completion.wait_with_progress().await;
        let progress_count = progress.len();
        self.progress.extend(progress);
        let success = result.map_err(|error| {
            MeshingStageExecutionError::new(format!(
                "remote {:?} partition {}/{} failed after {progress_count} progress records: {error}",
                stage.host().workload.stage,
                stage.host().workload.partition.partition_index,
                stage.host().workload.partition.partition_count,
            ))
        })?;
        let Some(invocation) = reused_invocation else {
            return Ok(success);
        };
        let mut identity = b"runmat-accepted-meshing-invocation/v1\0".to_vec();
        identity.extend_from_slice(invocation.bytes());
        let identity = StableDigest::from_bytes(*Digest::sha256(identity).bytes());
        let source = Arc::clone(&self.pool);
        let inventory_limits = self.inventory_limits;
        // CacheExport skips objects already held by the pool, so only the new
        // observation is written; the accepted numerical payload is not copied.
        let rebound = runmat_meshing_execution::rebind_committed_meshing_stage(
            source.as_ref(),
            self,
            stage.host(),
            &success,
            identity,
            inventory_limits,
        )
        .map_err(stage_error)?;
        // Publish locally derived observations into the pool before exposing
        // their references. A terminal stage has no later input-registration
        // phase, and its executor-local store does not outlive the DAG.
        for reference in rebound.result_objects() {
            self.register_reference(reference)?;
            self.pending.remove(&reference.logical_digest);
        }
        Ok(rebound)
    }
}

impl CacheImport for RemoteExactMeshingExecutor {
    fn read_verified_bounded_with_admission(
        &self,
        digest: Digest,
        maximum_bytes: u64,
        admit: &mut dyn FnMut(u64) -> ArtifactResult<()>,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        if let Some(bytes) =
            self.pool
                .read_verified_bounded_with_admission(digest, maximum_bytes, admit)?
        {
            return Ok(Some(bytes));
        }
        self.local_store
            .read_verified_bounded_with_admission(digest, maximum_bytes, admit)
    }
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        if let Some(bytes) = self.pool.read_verified_bounded(digest, maximum_bytes)? {
            return Ok(Some(bytes));
        }
        self.local_store
            .read_verified_bounded(digest, maximum_bytes)
    }
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.read_verified_bounded(digest, u64::MAX)
    }
}

impl CacheExport for RemoteExactMeshingExecutor {
    fn write_verified(&mut self, object: &LogicalObject) -> ArtifactResult<()> {
        if self
            .pool
            .execution_object_by_digest(object.descriptor.digest)
            .map_err(|error| runmat_execution_artifact::ArtifactError::Invalid(error.to_string()))?
            .is_some()
        {
            return Ok(());
        }
        self.local_store.write_verified(object)?;
        let reference = self
            .artifact_access
            .reference(
                object,
                ValueRefKind::DriverObject,
                object.descriptor.logical_name.clone(),
            )
            .map_err(|error| {
                runmat_execution_artifact::ArtifactError::Invalid(error.to_string())
            })?;
        self.pending.insert(reference.logical_digest, reference);
        Ok(())
    }
}

impl ExactMeshingDagExecutor for RemoteExactMeshingExecutor {
    async fn execute_stage<'a>(
        &'a mut self,
        stage: &'a PlannedMeshingStage,
        revision: ProgramRevision,
    ) -> Result<AttemptSuccess, MeshingStageExecutionError> {
        let planned = runmat_meshing_execution::meshing_task_id(self.pool.scope_id(), stage.host())
            .map_err(stage_error)?;
        let mut wave = crate::remote::RemoteObservedWave::begin(Arc::clone(&self.pool), &[planned])
            .await
            .map_err(stage_error)?;
        let completion = self.submit_stage(stage, revision, &mut wave).await?;
        wave.checkpoint().await.map_err(stage_error)?;
        let success = self.finish(completion, stage).await?;
        wave.close().await.map_err(stage_error)?;
        Ok(success)
    }

    async fn execute_stages<'a>(
        &'a mut self,
        stages: &'a [PlannedMeshingStage],
        revision: ProgramRevision,
    ) -> Result<Vec<AttemptSuccess>, MeshingStageExecutionError> {
        if stages.is_empty() {
            return Ok(Vec::new());
        }
        let planned = stages
            .iter()
            .map(|stage| {
                runmat_meshing_execution::meshing_task_id(self.pool.scope_id(), stage.host())
                    .map_err(stage_error)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut wave = crate::remote::RemoteObservedWave::begin(Arc::clone(&self.pool), &planned)
            .await
            .map_err(stage_error)?;
        let mut prepared = Vec::with_capacity(stages.len());
        for stage in stages {
            let program = self.prepare_stage(stage, revision.clone())?;
            let submission = self.submission(stage, &program)?;
            prepared.push((program, submission));
        }
        let task_ids = prepared
            .iter()
            .map(|(_, submission)| submission.request.id)
            .collect::<Vec<_>>();
        wave.arm(&task_ids).map_err(stage_error)?;
        let completions = self.pool.submit_graph(prepared).map_err(stage_error)?;
        wave.checkpoint().await.map_err(stage_error)?;
        let mut successes = Vec::with_capacity(completions.len());
        for (completion, stage) in completions.into_iter().zip(stages) {
            successes.push(self.finish(completion, stage).await?);
        }
        wave.close().await.map_err(stage_error)?;
        Ok(successes)
    }
}

fn stage_error(error: impl std::fmt::Display) -> MeshingStageExecutionError {
    MeshingStageExecutionError::new(error.to_string())
}
