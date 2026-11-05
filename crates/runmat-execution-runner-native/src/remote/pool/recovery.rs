//! Portable pool authority. Object bytes are streamed separately from bounded metadata.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use runmat_execution::identity::{AttemptId, WorkerId};
use runmat_execution::security::{
    EndpointTrustPolicy, NodeClockEndV1, NodeClockStartV1, WorkerAttemptJournalEntryV1,
    WorkerTreeRetirementEvidenceV1,
};
use runmat_execution::value::{ValueLimits, ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::{Digest, ExecutionScopeId, PoolId, TaskId};
use runmat_execution_artifact::encryption::{EncryptionPurpose, RunKeyMaterial};
use runmat_execution_artifact::ProgramExecutionRequest;
use runmat_execution_runner::recovery::DriverCheckpoint;
use runmat_execution_runner::PoolSpec;
use serde::{Deserialize, Serialize};

use super::RemotePoolDriver;
use crate::remote::clock_observer::{CertifiedWave, RemoteWaveIdentity};
use crate::{NativeExecutionError, NativeExecutionResult};

const MAX_CHECKPOINT_METADATA_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TaskBinding {
    program: ProgramExecutionRequest,
    inventory: Option<Vec<ValueRef>>,
}

impl RemotePoolCheckpoint {
    #[cfg(all(test, feature = "occt-native"))]
    pub(crate) fn test_completed_task_wall_millis(&self, task: TaskId) -> Option<u64> {
        self.completed_task_wall_millis.get(&task).copied()
    }

    pub(crate) fn closed_clock_release_entries(
        &self,
    ) -> NativeExecutionResult<
        Vec<runmat_execution_transport_native::control::ClockObservationReleaseEntryV1>,
    > {
        self.clock_ends
            .iter()
            .map(|(id, end)| {
                let start = self
                    .clock_starts
                    .get(id)
                    .ok_or_else(|| invalid("restored closed observer lacks its original start"))?;
                end.measured_wall_millis(start).map_err(invalid)?;
                Ok(
                    runmat_execution_transport_native::control::ClockObservationReleaseEntryV1 {
                        observation_id: *id,
                        start_receipt_digest: start.receipt_digest().map_err(invalid)?,
                        end_receipt_digest: end.receipt_digest().map_err(invalid)?,
                    },
                )
            })
            .collect()
    }

    pub(crate) fn prune_released_clocks(
        &mut self,
        entries: &[runmat_execution_transport_native::control::ClockObservationReleaseEntryV1],
    ) -> NativeExecutionResult<()> {
        let mut certified = Vec::with_capacity(entries.len());
        for entry in entries {
            let start = self
                .clock_starts
                .get(&entry.observation_id)
                .ok_or_else(|| invalid("restored released clock start is missing"))?;
            let end = self
                .clock_ends
                .get(&entry.observation_id)
                .ok_or_else(|| invalid("restored released clock end is missing"))?;
            if start.receipt_digest().map_err(invalid)? != entry.start_receipt_digest
                || end.receipt_digest().map_err(invalid)? != entry.end_receipt_digest
                || self.task_clocks.iter().any(|(task, id)| {
                    *id == entry.observation_id
                        && !self.completed_task_wall_millis.contains_key(task)
                })
            {
                return Err(invalid(
                    "restored release differs from finalized signed wall accounting",
                ));
            }
            let wall = end.measured_wall_millis(start).map_err(invalid)?;
            let mut tasks = self
                .task_clocks
                .iter()
                .filter_map(|(task, id)| (*id == entry.observation_id).then_some(*task))
                .collect::<Vec<_>>();
            tasks.sort_unstable();
            let wave = RemoteWaveIdentity::new(self.scope_id, &tasks)?;
            if self.wave_clocks.get(&wave) != Some(&entry.observation_id)
                || tasks
                    .iter()
                    .any(|task| self.completed_task_wall_millis.get(task) != Some(&wall))
                || self.certified_waves.contains_key(&wave)
                || certified.iter().any(|(prior, _)| *prior == wave)
            {
                return Err(invalid(
                    "restored release differs from original certified wave",
                ));
            }
            certified.push((
                wave,
                CertifiedWave {
                    tasks,
                    measured_wall_millis: wall,
                },
            ));
        }
        self.certified_waves.extend(certified);
        for entry in entries {
            self.clock_starts.remove(&entry.observation_id);
            self.clock_ends.remove(&entry.observation_id);
            self.task_clocks.retain(|_, id| *id != entry.observation_id);
            self.wave_clocks.retain(|_, id| *id != entry.observation_id);
        }
        Ok(())
    }
}

/// Bounded by active physical workers, recorded only after endpoint signature
/// verification and exact allocation admission. The checkpoint is itself
/// carried by the authenticated encrypted run-scoped recovery closure.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(in crate::remote) struct OriginalWorkerOwner {
    pub(in crate::remote) allocation_lease_id: String,
    pub(in crate::remote) allocation_fencing_token: u64,
    pub(in crate::remote) driver_fencing_token: u64,
    pub(in crate::remote) node_id: String,
    pub(in crate::remote) cluster_id: String,
    pub(in crate::remote) identity_fingerprint: String,
}

/// Issued only while decoding the pool catalog named by the authenticated
/// latest accepted Server checkpoint. A directly restored pool has no such
/// authority to turn absence from a later journal into non-admission proof.
pub(in crate::remote) struct AuthoritativeUnstartedInventory {
    pool_id: PoolId,
    source_driver_fence: u64,
    requests: BTreeMap<AttemptId, (WorkerId, Digest)>,
}

impl AuthoritativeUnstartedInventory {
    fn matches(&self, pool_id: PoolId, request: &runmat_execution_runner::AttemptRequest) -> bool {
        self.pool_id == pool_id
            && request.driver_fence <= self.source_driver_fence
            && self
                .requests
                .get(&request.id)
                .is_some_and(|(worker, digest)| {
                    *worker == request.worker_id
                        && Digest::sha256_serialized(request).ok().as_ref() == Some(digest)
                })
    }
}

/// A private negative-admission witness, distinct from process-tree cessation.
/// Only this module may construct it after validating a complete original
/// node-owned retired journal against the latest accepted checkpoint.
pub(super) struct CertifiedNeverAdmitted {
    request: runmat_execution_runner::AttemptRequest,
}

impl CertifiedNeverAdmitted {
    pub(super) fn request(&self) -> &runmat_execution_runner::AttemptRequest {
        &self.request
    }

    pub(super) fn matches(&self, request: &runmat_execution_runner::AttemptRequest) -> bool {
        &self.request == request
    }
}

/// Authenticated transport must bind these bytes to the run/request and checkpoint sequence.
/// This manifest validates consistency; it does not authenticate itself.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RemotePoolCheckpoint {
    schema_version: u16,
    scope_id: ExecutionScopeId,
    pool_id: PoolId,
    value_scope: String,
    bundle_digest: Digest,
    scheduler: DriverCheckpoint,
    #[serde(default)]
    original_owners: BTreeMap<WorkerId, OriginalWorkerOwner>,
    tasks: BTreeMap<TaskId, TaskBinding>,
    /// Original host-observed intervals for committed tasks, not scheduler logical clocks.
    completed_task_wall_millis: BTreeMap<TaskId, u64>,
    #[serde(
        serialize_with = "clock_codec::serialize",
        deserialize_with = "clock_codec::deserialize_clock"
    )]
    clock_starts: BTreeMap<[u8; 16], NodeClockStartV1>,
    #[serde(
        serialize_with = "clock_codec::serialize",
        deserialize_with = "clock_codec::deserialize_clock"
    )]
    clock_ends: BTreeMap<[u8; 16], NodeClockEndV1>,
    task_clocks: BTreeMap<TaskId, [u8; 16]>,
    #[serde(
        serialize_with = "clock_codec::serialize",
        deserialize_with = "clock_codec::deserialize_clock"
    )]
    wave_clocks: BTreeMap<RemoteWaveIdentity, [u8; 16]>,
    #[serde(
        serialize_with = "clock_codec::serialize",
        deserialize_with = "clock_codec::deserialize_metadata"
    )]
    certified_waves: BTreeMap<RemoteWaveIdentity, CertifiedWave>,
    execution_inputs: Vec<ValueRef>,
    value_inputs: Vec<ValueRef>,
}

mod clock_codec;

/// Replacement authority supplied by the authenticated allocation, never by the checkpoint.
pub struct RemotePoolRestoreAuthority {
    pub scope_id: ExecutionScopeId,
    pub pool: PoolSpec,
    pub value_scope: String,
    pub driver_fence: u64,
}

impl RemotePoolCheckpoint {
    pub(in crate::remote) fn authoritative_unstarted_inventory(
        &self,
    ) -> NativeExecutionResult<AuthoritativeUnstartedInventory> {
        self.validate()?;
        let source_driver_fence = self.scheduler.snapshot.driver_fence;
        let requests = self
            .scheduler
            .snapshot
            .attempts
            .values()
            .filter(|attempt| {
                matches!(
                    attempt.state,
                    runmat_execution::state::AttemptState::Assigned
                        | runmat_execution::state::AttemptState::Starting
                ) && attempt.terminal.is_none()
            })
            .map(|attempt| {
                if attempt.request.driver_fence > source_driver_fence
                    || !self
                        .original_owners
                        .contains_key(&attempt.request.worker_id)
                {
                    return Err(invalid(
                        "unstarted candidate lacks its original checkpoint owner",
                    ));
                }
                Ok((
                    attempt.request.id,
                    (
                        attempt.request.worker_id,
                        Digest::sha256_serialized(&attempt.request).map_err(invalid)?,
                    ),
                ))
            })
            .collect::<NativeExecutionResult<BTreeMap<_, _>>>()?;
        Ok(AuthoritativeUnstartedInventory {
            pool_id: self.pool_id,
            source_driver_fence,
            requests,
        })
    }

    pub(crate) fn scheduler_event_watermark(&self) -> u64 {
        self.scheduler.snapshot.next_event_sequence
    }

    /// Only terminal, accounted attempts in this exact snapshot may relinquish
    /// their worker-sandbox journal after the enclosing driver checkpoint is
    /// acknowledged by the authoritative store.
    pub(crate) fn checkpointed_attempts(
        &self,
    ) -> NativeExecutionResult<Vec<(WorkerId, super::super::CheckpointedAttempt)>> {
        self.scheduler
            .snapshot
            .attempts
            .values()
            .filter(|attempt| {
                attempt
                    .terminal
                    .as_ref()
                    .is_some_and(|terminal| terminal.resources.is_some())
                    && attempt
                        .unverified_retirement
                        .as_ref()
                        .is_none_or(|record| record.retired_at_driver_millis.is_some())
            })
            .map(|attempt| {
                Ok((
                    attempt.request.worker_id,
                    super::super::CheckpointedAttempt {
                        attempt_id: attempt.request.id,
                        request_digest: Digest::sha256_serialized(&attempt.request)
                            .map_err(invalid)?,
                    },
                ))
            })
            .collect()
    }

    #[cfg(all(test, feature = "occt-native"))]
    pub(crate) fn test_snapshot(&self) -> &runmat_execution_runner::DriverSnapshot {
        &self.scheduler.snapshot
    }

    #[cfg(all(test, feature = "occt-native"))]
    pub(crate) fn test_program(&self, task_id: TaskId) -> Option<&ProgramExecutionRequest> {
        self.tasks.get(&task_id).map(|binding| &binding.program)
    }

    pub fn encode(&self) -> NativeExecutionResult<Vec<u8>> {
        self.validate()?;
        let mut writer = BoundedMetadata(Vec::new());
        serde_json::to_writer(&mut writer, self).map_err(invalid)?;
        Ok(writer.0)
    }

    pub fn decode(bytes: &[u8]) -> NativeExecutionResult<Self> {
        if bytes.len() > MAX_CHECKPOINT_METADATA_BYTES {
            return Err(invalid("remote checkpoint metadata exceeds its limit"));
        }
        let checkpoint: Self = serde_json::from_slice(bytes).map_err(invalid)?;
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    /// Stable references only: uncommitted attempt downloads are deliberately absent.
    pub fn object_references(&self) -> NativeExecutionResult<Vec<ValueRef>> {
        let mut references = BTreeMap::new();
        for reference in self
            .execution_inputs
            .iter()
            .chain(&self.value_inputs)
            .chain(
                self.scheduler
                    .snapshot
                    .tasks
                    .values()
                    .filter_map(|task| task.committed.as_ref())
                    .flat_map(|commit| commit.result.result_objects()),
            )
        {
            if reference.authorization_scope != self.value_scope {
                return Err(invalid("foreign checkpoint object scope"));
            }
            if !matches!(
                reference.kind,
                ValueRefKind::DriverObject | ValueRefKind::ResultObject
            ) || reference.resident_fence.is_some()
            {
                return Err(invalid("checkpoint object is not portable"));
            }
            ValuePayload::Object(Box::new(reference.clone()))
                .validate(ValueLimits::default())
                .map_err(invalid)?;
            if reference.encoded_length == 0
                || reference.encoded_length
                    > runmat_execution_artifact::MAX_PROGRAM_EXECUTION_OBJECT_BYTES
            {
                return Err(invalid("checkpoint object exceeds its byte limit"));
            }
            if let Some(previous) = references.get(&reference.id) {
                if !same_object(previous, reference) {
                    return Err(invalid("conflicting checkpoint object references"));
                }
            } else {
                references.insert(reference.id, reference.clone());
            }
        }
        if references.len() > super::super::object_transfer::MAX_REMOTE_OBJECTS {
            return Err(invalid(
                "remote checkpoint object inventory exceeds its limit",
            ));
        }
        let total = references
            .values()
            .try_fold(0_u64, |total, reference| {
                total.checked_add(reference.encoded_length)
            })
            .ok_or_else(|| invalid("checkpoint object byte total overflow"))?;
        if total > runmat_execution_artifact::MAX_PROGRAM_EXECUTION_OBJECT_TOTAL_BYTES {
            return Err(invalid("checkpoint object byte total exceeds its limit"));
        }
        Ok(references.into_values().collect())
    }

    fn validate(&self) -> NativeExecutionResult<()> {
        if self.schema_version != 6 {
            return Err(invalid("unsupported remote checkpoint schema"));
        }
        self.scheduler.validate()?;
        let snapshot = &self.scheduler.snapshot;
        if self.clock_starts.len() > runmat_execution::security::MAX_CLOCK_OBSERVATIONS_PER_NODE
            || self.clock_ends.len() > self.clock_starts.len()
            || self.task_clocks.len() > snapshot.tasks.len()
            || self.wave_clocks.len() != self.clock_starts.len()
        {
            return Err(invalid(
                "remote checkpoint clock ownership exceeds its bounded scope",
            ));
        }
        for (id, start) in &self.clock_starts {
            start.validate_shape().map_err(invalid)?;
            if *id != start.owner.observation_id || start.owner.run_id != self.value_scope {
                return Err(invalid("remote checkpoint clock start has foreign owner"));
            }
            let policy = EndpointTrustPolicy {
                permitted_tiers: Default::default(),
                trusted_identity_fingerprints: [start.owner.identity_fingerprint.clone()]
                    .into_iter()
                    .collect(),
                allowed_attestation_classes: Default::default(),
                require_pinned_identity: true,
                now_unix_millis: 0,
                maximum_clock_skew_millis: 0,
            };
            policy
                .verify_node_clock_start(start, &start.owner.identity_fingerprint)
                .map_err(invalid)?;
        }
        for (id, end) in &self.clock_ends {
            let start = self
                .clock_starts
                .get(id)
                .ok_or_else(|| invalid("remote checkpoint clock end has no start"))?;
            end.measured_wall_millis(start).map_err(invalid)?;
            let policy = EndpointTrustPolicy {
                permitted_tiers: Default::default(),
                trusted_identity_fingerprints: [start.owner.identity_fingerprint.clone()]
                    .into_iter()
                    .collect(),
                allowed_attestation_classes: Default::default(),
                require_pinned_identity: true,
                now_unix_millis: 0,
                maximum_clock_skew_millis: 0,
            };
            policy
                .verify_node_clock_end(end, &start.owner.identity_fingerprint)
                .map_err(invalid)?;
        }
        for (task, id) in &self.task_clocks {
            if !snapshot.tasks.contains_key(task) || !self.clock_starts.contains_key(id) {
                return Err(invalid("remote checkpoint task has foreign wave observer"));
            }
            if let Some(end) = self.clock_ends.get(id) {
                let wall = end
                    .measured_wall_millis(&self.clock_starts[id])
                    .map_err(invalid)?;
                if self
                    .completed_task_wall_millis
                    .get(task)
                    .is_some_and(|saved| *saved != wall)
                {
                    return Err(invalid(
                        "remote checkpoint wall differs from signed observer",
                    ));
                }
            }
        }
        for (wave, id) in &self.wave_clocks {
            if !self.clock_starts.contains_key(id)
                || self
                    .wave_clocks
                    .values()
                    .filter(|other| *other == id)
                    .count()
                    != 1
            {
                return Err(invalid(
                    "remote checkpoint wave has no unique original observer",
                ));
            }
            let tasks = self
                .task_clocks
                .iter()
                .filter_map(|(task, owner)| (*owner == *id).then_some(*task))
                .collect::<Vec<_>>();
            if !tasks.is_empty() && RemoteWaveIdentity::new(self.scope_id, &tasks)? != *wave {
                return Err(invalid(
                    "remote checkpoint task inventory differs from original wave",
                ));
            }
        }
        let mut certified_tasks = BTreeSet::new();
        for (wave, certified) in &self.certified_waves {
            if certified.measured_wall_millis == 0
                || self.wave_clocks.contains_key(wave)
                || RemoteWaveIdentity::new(self.scope_id, &certified.tasks)? != *wave
                || certified.tasks.iter().any(|task| {
                    !certified_tasks.insert(*task)
                        || self.task_clocks.contains_key(task)
                        || self.completed_task_wall_millis.get(task)
                            != Some(&certified.measured_wall_millis)
                        || !snapshot.tasks.get(task).is_some_and(|record| {
                            record.state == runmat_execution::state::TaskState::Succeeded
                        })
                })
            {
                return Err(invalid(
                    "certified wave differs from exact committed task inventory",
                ));
            }
        }
        if self.completed_task_wall_millis.iter().any(|(task, wall)| {
            *wall == 0
                || !snapshot.tasks.get(task).is_some_and(|record| {
                    record.state == runmat_execution::state::TaskState::Succeeded
                })
        }) || snapshot.tasks.iter().any(|(task, record)| {
            record.state == runmat_execution::state::TaskState::Succeeded
                && !self.completed_task_wall_millis.contains_key(task)
                && !self
                    .task_clocks
                    .get(task)
                    .is_some_and(|id| self.clock_starts.contains_key(id))
        }) {
            return Err(invalid("committed task lacks original host wall interval"));
        }
        for (task, wall) in &self.completed_task_wall_millis {
            snapshot
                .task_execution_evidence(*task)
                .map_err(invalid)?
                .resource_aggregate(*wall)
                .map_err(invalid)?;
        }
        if snapshot.pools.len() != 1
            || !snapshot.pools.contains_key(&self.pool_id)
            || self.tasks.len() != snapshot.tasks.len()
        {
            return Err(invalid(
                "remote checkpoint task or pool authority is incomplete",
            ));
        }
        let pool = &snapshot.pools[&self.pool_id];
        if self.original_owners.len() > pool.spec.max_workers as usize {
            return Err(invalid(
                "checkpoint original ownership exceeds physical worker capacity",
            ));
        }
        for (worker_id, owner) in &self.original_owners {
            let derived = WorkerId::derive(&[
                self.value_scope.as_bytes(),
                owner.allocation_lease_id.as_bytes(),
            ]);
            if *worker_id != derived
                || owner.allocation_fencing_token == 0
                || owner.driver_fencing_token == 0
                || owner.allocation_lease_id.is_empty()
                || owner.node_id.is_empty()
                || owner.cluster_id.is_empty()
                || owner.identity_fingerprint.len() != 64
                || !pool.workers.contains_key(worker_id)
            {
                return Err(invalid(
                    "checkpoint original worker ownership is malformed or foreign",
                ));
            }
        }
        let objects = self.object_references()?;
        for reference in self.execution_inputs.iter().chain(
            snapshot
                .tasks
                .values()
                .filter_map(|task| task.committed.as_ref())
                .flat_map(|commit| commit.result.result_objects()),
        ) {
            if !matches!(
                reference.kind,
                ValueRefKind::DriverObject | ValueRefKind::ResultObject
            ) {
                return Err(invalid("checkpoint execution object is not portable"));
            }
        }
        let mut inventories = super::input_inventory::InputInventories::default();
        for (id, task) in &snapshot.tasks {
            let binding = self
                .tasks
                .get(id)
                .ok_or_else(|| invalid("checkpoint program missing"))?;
            let request = &task.submission.request;
            let program = &binding.program;
            program.validate().map_err(invalid)?;
            if request.scope_id != self.scope_id
                || request.pool_id != self.pool_id
                || request.program_artifact_id()
                    != Some(runmat_execution::identity::ArtifactId::derive(&[program
                        .artifact
                        .id
                        .0
                        .bytes()]))
                || request.callable.program != program.callable
                || request.invocation_context != program.context
                || request.environment != program.environment
                || request.console != program.console
                || request.inputs != program.arguments
                || request.outputs.requested_outputs != program.requested_outputs
            {
                return Err(invalid("checkpoint task differs from its exact program"));
            }
            if let Some(inventory) = &binding.inventory {
                if inventory
                    .iter()
                    .any(|reference| !objects.iter().any(|stored| same_object(stored, reference)))
                {
                    return Err(invalid("checkpoint omits an admitted input object"));
                }
                inventories.register(program, inventory.clone())?;
            }
            inventories.bind(*id, program)?;
            if let Some(commit) = &task.committed {
                let (outputs, _) = commit
                    .result
                    .values()
                    .ok_or_else(|| invalid("checkpoint commit is not a portable value result"))?;
                for output in outputs {
                    output.validate(ValueLimits::default()).map_err(invalid)?;
                }
                if super::super::value_transfer::collect_references(outputs)
                    .iter()
                    .any(|reference| !objects.iter().any(|stored| same_object(stored, reference)))
                {
                    return Err(invalid("checkpoint omits a committed output object"));
                }
            }
            for reference in super::super::value_transfer::collect_references(&program.arguments) {
                if !objects.iter().any(|stored| same_object(stored, &reference)) {
                    return Err(invalid("checkpoint omits a program argument object"));
                }
            }
        }
        Ok(())
    }
}

impl RemotePoolDriver {
    pub(in crate::remote) async fn prune_durable_attempt_journals(
        &self,
        sequence: u64,
        attempts: &[(WorkerId, super::super::CheckpointedAttempt)],
    ) {
        let mut by_worker = BTreeMap::<WorkerId, Vec<super::super::CheckpointedAttempt>>::new();
        for (worker_id, attempt) in attempts {
            by_worker.entry(*worker_id).or_default().push(*attempt);
        }
        let routes = self.channels.read().await;
        let pending = by_worker
            .into_iter()
            .filter_map(|(worker_id, attempts)| {
                routes
                    .get(&worker_id)
                    .map(|route| (worker_id, Arc::clone(&route.channel), attempts))
            })
            .collect::<Vec<_>>();
        drop(routes);
        for (worker_id, channel, attempts) in pending {
            for batch in attempts.chunks(runmat_execution::security::MAX_RETIREMENT_ATTEMPTS) {
                if let Err(error) = channel.prune_durable_checkpoint(sequence, batch).await {
                    tracing::warn!(%worker_id, %error, "worker retained checkpointed attempt journal");
                    break;
                }
            }
        }
    }

    pub(in crate::remote) fn unresolved_original_owners(
        &self,
    ) -> NativeExecutionResult<BTreeMap<WorkerId, OriginalWorkerOwner>> {
        let requests = {
            let driver = self.driver.lock().expect("remote driver poisoned");
            driver
                .snapshot_ref()
                .attempts
                .values()
                .filter(|attempt| {
                    attempt
                        .unverified_retirement
                        .as_ref()
                        .is_some_and(|record| record.retired_at_driver_millis.is_none())
                })
                .map(|attempt| (attempt.request.worker_id, attempt.request.driver_fence))
                .collect::<Vec<_>>()
        };
        let owners = self
            .original_owners
            .lock()
            .expect("original owners poisoned");
        let mut unresolved = BTreeMap::new();
        for (worker_id, driver_fence) in requests {
            let owner = owners.get(&worker_id).ok_or_else(|| {
                invalid("uncertain original worker lacks authenticated allocation owner")
            })?;
            if driver_fence != owner.driver_fencing_token {
                return Err(invalid(
                    "uncertain attempt differs from its original driver fence",
                ));
            }
            unresolved.insert(worker_id, owner.clone());
        }
        Ok(unresolved)
    }

    pub(in crate::remote) fn accept_node_tree_retirement(
        self: &Arc<Self>,
        authority: &runmat_execution_transport_native::control::DriverAuthority,
        worker_id: WorkerId,
        owner: &OriginalWorkerOwner,
        evidence: &WorkerTreeRetirementEvidenceV1,
        run_key: &RunKeyMaterial,
        authoritative_unstarted: Option<&AuthoritativeUnstartedInventory>,
    ) -> NativeExecutionResult<()> {
        if self.value_scope != authority.run_id
            || evidence.org_id != authority.org_id
            || evidence.cluster_id != owner.cluster_id
            || evidence.node_id != owner.node_id
            || evidence.run_id != authority.run_id
            || evidence.allocation_lease_id != owner.allocation_lease_id
            || evidence.allocation_fencing_token != owner.allocation_fencing_token
            || evidence.driver_fencing_token != owner.driver_fencing_token
            || owner.driver_fencing_token >= authority.fencing_token
            || worker_id
                != WorkerId::derive(&[
                    authority.run_id.as_bytes(),
                    owner.allocation_lease_id.as_bytes(),
                ])
        {
            return Err(invalid(
                "node retirement certificate differs from original worker allocation or fence",
            ));
        }
        let now_unix_millis = u64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(invalid)?
                .as_millis(),
        )
        .map_err(|_| invalid("system time exceeds retirement evidence range"))?;
        let policy = EndpointTrustPolicy {
            permitted_tiers: Default::default(),
            trusted_identity_fingerprints: [owner.identity_fingerprint.clone()]
                .into_iter()
                .collect(),
            allowed_attestation_classes: Default::default(),
            require_pinned_identity: true,
            now_unix_millis,
            maximum_clock_skew_millis: 30_000,
        };
        policy
            .verify_worker_tree_retirement(evidence, &owner.identity_fingerprint)
            .map_err(invalid)?;
        let mut claims = BTreeMap::new();
        for encoded in &evidence.encrypted_attempt_receipts {
            let bytes = URL_SAFE_NO_PAD.decode(encoded).map_err(invalid)?;
            let plaintext = super::super::crypto::open_object(
                run_key,
                &bytes,
                &authority.run_id,
                EncryptionPurpose::RetirementReceipt,
            )?;
            if plaintext.len() < 33 {
                return Err(invalid(
                    "encrypted retirement receipt lacks private salt or journal",
                ));
            }
            let entry: WorkerAttemptJournalEntryV1 =
                serde_json::from_slice(&plaintext[32..]).map_err(invalid)?;
            entry.validate_shape().map_err(invalid)?;
            if entry.run_id != authority.run_id
                || entry.allocation_lease_id != owner.allocation_lease_id
                || entry.allocation_fencing_token != owner.allocation_fencing_token
                || entry.driver_fencing_token != owner.driver_fencing_token
                || entry.process_launch_nonce != evidence.process_launch_nonce
                || entry.worker_id != worker_id
                || entry.clock_domain_nonce != evidence.clock_domain_nonce
                || entry.ticks_per_second != evidence.ticks_per_second
                || entry.started_monotonic_ticks >= evidence.retired_monotonic_ticks
                || claims.contains_key(&entry.attempt_id)
            {
                return Err(invalid(
                    "encrypted retirement receipt differs from original owner",
                ));
            }
            let claim = entry
                .into_retired_claim(evidence.retired_monotonic_ticks)
                .map_err(invalid)?;
            claims.insert(claim.attempt_id, claim);
        }
        let requests = {
            let driver = self.driver.lock().expect("remote driver poisoned");
            let snapshot = driver.snapshot_ref();
            for claim in claims.values() {
                let original = snapshot
                    .attempts
                    .values()
                    .find(|attempt| attempt.request.id == claim.attempt_id)
                    .ok_or_else(|| {
                        invalid("retirement receipt has no original checkpoint attempt")
                    })?;
                let request = &original.request;
                if request.worker_id != worker_id
                    || request.driver_fence != owner.driver_fencing_token
                    || claim.task_id != request.task_id
                    || claim.worker_id != request.worker_id
                    || claim.ordinal != request.ordinal
                    || claim.driver_fencing_token != request.driver_fence
                    || claim.request_digest
                        != Digest::sha256_serialized(request).map_err(invalid)?
                {
                    return Err(invalid(
                        "retirement receipt differs from original checkpoint assignment",
                    ));
                }
            }
            snapshot
                .attempts
                .values()
                .filter(|attempt| {
                    attempt.request.worker_id == worker_id
                        && attempt.request.driver_fence == owner.driver_fencing_token
                        && attempt
                            .unverified_retirement
                            .as_ref()
                            .is_some_and(|record| record.retired_at_driver_millis.is_none())
                })
                .map(|attempt| {
                    (
                        attempt.request.clone(),
                        attempt.state,
                        attempt.terminal.is_some(),
                        attempt
                            .unverified_retirement
                            .as_ref()
                            .is_some_and(|record| {
                                record.resources.is_none()
                                    && record.execution.is_none()
                                    && record.console.is_none()
                                    && record.console_scopes.is_empty()
                                    && record.failure.is_none()
                                    && record.failure_kind.is_none()
                                    && record.retired_at_driver_millis.is_none()
                            }),
                    )
                })
                .collect::<Vec<_>>()
        };
        if requests.is_empty() {
            return Err(invalid(
                "node certificate has no uncertain original attempt",
            ));
        }
        enum RetirementDecision {
            Accounted(
                Box<(
                    runmat_execution_runner::AttemptRequest,
                    runmat_execution_runner::task::AttemptResourceAccounting,
                )>,
            ),
            NeverAdmitted(Box<CertifiedNeverAdmitted>),
        }
        const _: () =
            assert!(std::mem::size_of::<RetirementDecision>() <= 2 * std::mem::size_of::<usize>());
        let decisions = requests
            .into_iter()
            .map(|(request, state, has_terminal, unobserved)| {
                let Some(claim) = claims.get(&request.id) else {
                    // A complete original-owner journal is meaningful negative
                    // evidence only for an assignment in the latest accepted
                    // checkpoint. `begin` is fsynced before worker invocation;
                    // checkpoint ACK pruning rejects nonterminal entries.
                    if !authoritative_unstarted
                        .is_some_and(|inventory| inventory.matches(self.pool_id, &request))
                        || !matches!(
                            state,
                            runmat_execution::state::AttemptState::Assigned
                                | runmat_execution::state::AttemptState::Starting
                        )
                        || has_terminal || !unobserved
                    {
                        return Err(invalid(
                            "node retirement lacks original attempt accounting or certified non-admission",
                        ));
                    }
                    return Ok(RetirementDecision::NeverAdmitted(Box::new(CertifiedNeverAdmitted {
                        request,
                    })));
                };
                let request_digest = Digest::sha256_serialized(&request).map_err(invalid)?;
                if claim.task_id != request.task_id
                    || claim.worker_id != request.worker_id
                    || claim.ordinal != request.ordinal
                    || claim.driver_fencing_token != request.driver_fence
                    || claim.request_digest != request_digest
                {
                    return Err(invalid(
                        "node retirement accounting differs from original fenced assignment",
                    ));
                }
                let inventory = claim.published_result_inventory.as_ref();
                let resources =
                    runmat_execution_runner::task::AttemptResourceAccounting::for_attempt(
                        &request,
                        claim.measured_wall_millis().map_err(invalid)?,
                        inventory.map_or(0, |value| value.artifact_bytes),
                        inventory.map_or(0, |value| value.egress_bytes),
                        inventory.map_or(0, |value| value.relay_bytes),
                    )
                    .map_err(invalid)?;
                Ok(RetirementDecision::Accounted(Box::new((request, resources))))
            })
            .collect::<NativeExecutionResult<Vec<_>>>()?;
        for decision in decisions {
            let accepted = match decision {
                RetirementDecision::Accounted(accounted) => {
                    let (request, resources) = *accounted;
                    self.apply_authenticated_retirement(
                        &request,
                        super::super::AuthenticatedPhysicalRetirement::issue(request.clone()),
                        Some(resources),
                    )
                }
                RetirementDecision::NeverAdmitted(witness) => {
                    self.apply_certified_never_admitted(*witness)
                }
            };
            if !accepted {
                return Err(invalid(
                    "scheduler rejected exact original worker retirement",
                ));
            }
        }
        Ok(())
    }
    pub(in crate::remote) fn record_original_worker_owner(
        &self,
        worker_id: WorkerId,
        owner: OriginalWorkerOwner,
    ) -> NativeExecutionResult<()> {
        if worker_id
            != WorkerId::derive(&[
                self.value_scope.as_bytes(),
                owner.allocation_lease_id.as_bytes(),
            ])
            || owner.allocation_fencing_token == 0
            || owner.driver_fencing_token == 0
            || owner.identity_fingerprint.len() != 64
        {
            return Err(invalid(
                "authenticated worker owner differs from pool identity",
            ));
        }
        let pool = self
            .driver
            .lock()
            .expect("remote driver poisoned")
            .snapshot_ref()
            .pools[&self.pool_id]
            .clone();
        let mut owners = self
            .original_owners
            .lock()
            .expect("original owners poisoned");
        owners.retain(|id, _| {
            pool.workers.get(id).is_some_and(|worker| {
                worker.active_attempts > 0
                    || !matches!(
                        worker.lifecycle,
                        runmat_execution_runner::pool::WorkerLifecycle::Lost
                            | runmat_execution_runner::pool::WorkerLifecycle::Stopped
                    )
            })
        });
        if let Some(existing) = owners.get(&worker_id) {
            if existing != &owner {
                return Err(invalid(
                    "worker owner changed within the same allocation identity",
                ));
            }
        } else {
            if owners.len() >= pool.spec.max_workers as usize {
                return Err(invalid(
                    "authenticated worker owner journal exceeds pool capacity",
                ));
            }
            owners.insert(worker_id, owner);
        }
        Ok(())
    }
    /// Capture metadata under admission-compatible locks; no object IO occurs under these locks.
    pub fn capture_checkpoint(
        &self,
        generation: u64,
    ) -> NativeExecutionResult<RemotePoolCheckpoint> {
        let _clock_state = self.clock_state.lock().expect("clock state poisoned");
        let bundle_digest = self
            .direct_bundle
            .as_ref()
            .map(|bundle| bundle.digest)
            .ok_or_else(|| invalid("graph-only remote pools have no portable checkpoint"))?;
        let inventories = self
            .input_inventories
            .lock()
            .expect("input inventory registry poisoned");
        let programs = self
            .programs
            .lock()
            .expect("remote program catalog poisoned");
        let driver = self.driver.lock().expect("remote driver poisoned");
        let snapshot = driver.snapshot();
        let tasks = snapshot
            .tasks
            .keys()
            .map(|id| {
                let program = match programs.get(id).map(|program| program.as_ref()) {
                    Some(runmat_execution_artifact::WorkerProgramRequest::Direct(program)) => {
                        program
                    }
                    Some(runmat_execution_artifact::WorkerProgramRequest::Graph(_)) => {
                        return Err(invalid("graph programs have no portable checkpoint"))
                    }
                    None => return Err(invalid("admitted program missing during checkpoint")),
                };
                inventories.validate(*id, program)?;
                Ok((
                    *id,
                    TaskBinding {
                        program: program.as_ref().clone(),
                        inventory: inventories.get(*id),
                    },
                ))
            })
            .collect::<NativeExecutionResult<BTreeMap<_, _>>>()?;
        let mut execution_inputs = self.execution_objects.checkpoint_inputs();
        execution_inputs.sort_by_key(|reference| reference.id);
        let mut value_inputs = self.values.checkpoint_references();
        value_inputs.sort_by_key(|reference| reference.id);
        let original_owners = self
            .original_owners
            .lock()
            .expect("original owners poisoned")
            .iter()
            .filter(|(worker_id, _)| {
                snapshot.pools[&self.pool_id]
                    .workers
                    .get(worker_id)
                    .is_some_and(|worker| worker.active_attempts > 0)
            })
            .map(|(id, owner)| (*id, owner.clone()))
            .collect();
        let completed_task_wall_millis = super::timing::completed(
            &self
                .task_wall_intervals
                .lock()
                .expect("remote task wall intervals poisoned"),
        );
        let checkpoint = RemotePoolCheckpoint {
            schema_version: 6,
            scope_id: self.scope_id,
            pool_id: self.pool_id,
            value_scope: self.value_scope.clone(),
            bundle_digest,
            scheduler: DriverCheckpoint::new(generation, snapshot),
            original_owners,
            tasks,
            completed_task_wall_millis,
            clock_starts: self
                .clock_starts
                .lock()
                .expect("clock starts poisoned")
                .iter()
                .map(|(id, value)| (*id, value.clone()))
                .collect(),
            clock_ends: self
                .clock_ends
                .lock()
                .expect("clock ends poisoned")
                .iter()
                .map(|(id, value)| (*id, value.clone()))
                .collect(),
            task_clocks: self
                .task_clocks
                .lock()
                .expect("task clock bindings poisoned")
                .iter()
                .map(|(task, id)| (*task, *id))
                .collect(),
            wave_clocks: self
                .wave_clocks
                .lock()
                .expect("wave clocks poisoned")
                .iter()
                .map(|(wave, id)| (*wave, *id))
                .collect(),
            certified_waves: self
                .certified_waves
                .lock()
                .expect("certified waves poisoned")
                .iter()
                .map(|(wave, value)| (*wave, value.clone()))
                .collect(),
            execution_inputs,
            value_inputs,
        };
        drop(driver);
        drop(programs);
        drop(inventories);
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    /// Read one immutable object at a time after capture, without a scheduler lock.
    pub fn checkpoint_object(&self, reference: &ValueRef) -> NativeExecutionResult<Arc<[u8]>> {
        if let Some(bytes) = self.execution_objects.get(reference)? {
            return Ok(bytes);
        }
        self.values
            .encoded(reference)?
            .ok_or_else(|| invalid("checkpoint object is unavailable"))
    }

    /// Build an isolated pool. No workers, routes, transfer caches or waiter channels survive.
    /// All metadata and bytes must validate before the restored pool is exposed to its caller.
    pub fn restore_checkpoint(
        checkpoint: RemotePoolCheckpoint,
        authority: RemotePoolRestoreAuthority,
        bundle: impl Into<Arc<[u8]>>,
        mut read_object: impl FnMut(&ValueRef) -> NativeExecutionResult<Arc<[u8]>>,
    ) -> NativeExecutionResult<Arc<Self>> {
        checkpoint.validate()?;
        let bundle = bundle.into();
        if checkpoint.scope_id != authority.scope_id
            || checkpoint.pool_id != authority.pool.id
            || checkpoint.value_scope != authority.value_scope
            || checkpoint.bundle_digest != Digest::sha256(&bundle)
            || !same_allocation_authority(
                &checkpoint.scheduler.snapshot.pools[&checkpoint.pool_id].spec,
                &authority.pool,
            )
        {
            return Err(invalid(
                "checkpoint differs from replacement allocation authority",
            ));
        }
        let mut driver = checkpoint
            .scheduler
            .clone()
            .restore_at_fence(authority.driver_fence)?;
        // Reconciliation retains old attempts and reservations, but no worker is Ready.
        // Historic software proofs cannot authorize the replacement worker executable.
        driver.handle(
            runmat_execution_runner::DriverCommand::RefreshPoolCapabilities {
                pool_id: authority.pool.id,
            },
        )?;
        let mut pool = Self::new_with_value_scope(
            authority.scope_id,
            authority.pool,
            authority.driver_fence,
            bundle,
            authority.value_scope,
        )?;
        let restored = Arc::get_mut(&mut pool).expect("new pool is exclusively owned");
        *restored
            .original_owners
            .get_mut()
            .expect("new original owner map") = checkpoint
            .original_owners
            .iter()
            .map(|(id, owner)| (*id, owner.clone()))
            .collect();
        for reference in checkpoint.object_references()? {
            let bytes = read_object(&reference)?;
            if checkpoint.value_inputs.contains(&reference) {
                restored.values.register(
                    reference.clone(),
                    Arc::clone(&bytes),
                    &restored.value_scope,
                )?;
            }
            if checkpoint
                .execution_inputs
                .iter()
                .any(|stored| same_object(stored, &reference))
                || checkpoint
                    .scheduler
                    .snapshot
                    .tasks
                    .values()
                    .filter_map(|task| task.committed.as_ref())
                    .any(|commit| {
                        commit
                            .result
                            .result_objects()
                            .iter()
                            .any(|stored| same_object(stored, &reference))
                    })
            {
                restored.execution_objects.restore_object(
                    reference.clone(),
                    bytes,
                    &restored.value_scope,
                    checkpoint
                        .execution_inputs
                        .iter()
                        .any(|stored| same_object(stored, &reference)),
                )?;
            }
        }
        for task in checkpoint.scheduler.snapshot.tasks.values() {
            if let Some(commit) = &task.committed {
                restored
                    .execution_objects
                    .commit_results(commit.result.result_objects())?;
            }
        }
        let inventories = restored
            .input_inventories
            .get_mut()
            .expect("new inventory registry");
        let programs = restored.programs.get_mut().expect("new program catalog");
        for (id, binding) in checkpoint.tasks {
            if let Some(inventory) = binding.inventory {
                inventories.register(&binding.program, inventory)?;
            }
            inventories.bind(id, &binding.program)?;
            programs.insert(
                id,
                Arc::new(runmat_execution_artifact::WorkerProgramRequest::direct(
                    binding.program,
                )),
            );
        }
        *restored.driver.get_mut().expect("new driver") = driver;
        *restored
            .task_wall_intervals
            .get_mut()
            .expect("new task wall intervals") = checkpoint
            .completed_task_wall_millis
            .iter()
            .map(|(task, wall)| (*task, super::timing::TaskWallInterval::Completed(*wall)))
            .collect();
        *restored.clock_starts.get_mut().expect("new clock starts") = checkpoint
            .clock_starts
            .iter()
            .map(|(id, start)| (*id, start.clone()))
            .collect();
        *restored.clock_ends.get_mut().expect("new clock ends") = checkpoint
            .clock_ends
            .iter()
            .map(|(id, end)| (*id, end.clone()))
            .collect();
        *restored.task_clocks.get_mut().expect("new task clocks") = checkpoint
            .task_clocks
            .iter()
            .map(|(task, id)| (*task, *id))
            .collect();
        *restored.wave_clocks.get_mut().expect("new wave clocks") = checkpoint
            .wave_clocks
            .iter()
            .map(|(wave, id)| (*wave, *id))
            .collect();
        *restored
            .certified_waves
            .get_mut()
            .expect("new certified waves") = checkpoint
            .certified_waves
            .iter()
            .map(|(wave, value)| (*wave, value.clone()))
            .collect();
        for (task, id) in &checkpoint.task_clocks {
            if !checkpoint.completed_task_wall_millis.contains_key(task) {
                restored
                    .task_wall_intervals
                    .get_mut()
                    .expect("new wall intervals")
                    .insert(*task, super::timing::TaskWallInterval::ObservedPending(*id));
            }
        }
        // Install waiters before any worker can run. A completion arriving before
        // the composing procedure reattaches remains buffered in its original channel.
        for (id, task) in &checkpoint.scheduler.snapshot.tasks {
            if task.state == runmat_execution::state::TaskState::Succeeded {
                continue;
            }
            let (sender, receiver) = tokio::sync::oneshot::channel();
            let progress = Arc::new(super::super::pool_progress::RemoteProgressBuffer::default());
            restored
                .completions
                .get_mut()
                .expect("new completions")
                .insert(*id, sender);
            restored
                .progress
                .get_mut()
                .expect("new progress")
                .insert(*id, Arc::clone(&progress));
            restored
                .restored_completions
                .get_mut()
                .expect("new restored completions")
                .insert(
                    *id,
                    super::super::pool_progress::RemoteTaskCompletion::new(receiver, progress),
                );
        }
        // Reconciliation leaves no Ready worker. Expire original deadlines before
        // replacement worker registration is allowed to dispatch any task.
        pool.tick()?;
        Ok(pool)
    }
}

fn invalid(message: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(message.to_string())
}

fn same_allocation_authority(saved: &PoolSpec, replacement: &PoolSpec) -> bool {
    let mut saved = saved.clone();
    let mut replacement = replacement.clone();
    // Capabilities are a derived ceiling from authenticated Ready workers, not allocation
    // capacity. Preserve exact comparison of every physical resource and pool policy field.
    saved.resource_limit.capabilities.clear();
    replacement.resource_limit.capabilities.clear();
    saved == replacement
}

fn same_object(left: &ValueRef, right: &ValueRef) -> bool {
    left == right || super::super::object_transfer::references_share_content_identity(left, right)
}

struct BoundedMetadata(Vec<u8>);

impl std::io::Write for BoundedMetadata {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > MAX_CHECKPOINT_METADATA_BYTES.saturating_sub(self.0.len()) {
            return Err(std::io::Error::other(
                "remote checkpoint metadata exceeds its limit",
            ));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
