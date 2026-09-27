//! Portable pool authority. Object bytes are streamed separately from bounded metadata.
use std::collections::BTreeMap;
use std::sync::Arc;

use runmat_execution::value::{ValueLimits, ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::{Digest, ExecutionScopeId, PoolId, TaskId};
use runmat_execution_artifact::ProgramExecutionRequest;
use runmat_execution_runner::recovery::DriverCheckpoint;
use runmat_execution_runner::PoolSpec;
use serde::{Deserialize, Serialize};

use super::RemotePoolDriver;
use crate::{NativeExecutionError, NativeExecutionResult};

const MAX_CHECKPOINT_METADATA_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TaskBinding {
    program: ProgramExecutionRequest,
    inventory: Option<Vec<ValueRef>>,
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
    tasks: BTreeMap<TaskId, TaskBinding>,
    execution_inputs: Vec<ValueRef>,
    value_inputs: Vec<ValueRef>,
}

/// Replacement authority supplied by the authenticated allocation, never by the checkpoint.
pub struct RemotePoolRestoreAuthority {
    pub scope_id: ExecutionScopeId,
    pub pool: PoolSpec,
    pub value_scope: String,
    pub driver_fence: u64,
}

impl RemotePoolCheckpoint {
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
                    .flat_map(|commit| &commit.result_objects),
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
        if self.schema_version != 1 || self.scheduler.schema_version != 1 {
            return Err(invalid("unsupported remote checkpoint schema"));
        }
        self.scheduler.snapshot.validate()?;
        let snapshot = &self.scheduler.snapshot;
        if snapshot.pools.len() != 1
            || !snapshot.pools.contains_key(&self.pool_id)
            || self.tasks.len() != snapshot.tasks.len()
        {
            return Err(invalid(
                "remote checkpoint task or pool authority is incomplete",
            ));
        }
        let objects = self.object_references()?;
        for reference in self.execution_inputs.iter().chain(
            snapshot
                .tasks
                .values()
                .filter_map(|task| task.committed.as_ref())
                .flat_map(|commit| &commit.result_objects),
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
                || request.program_artifact_id
                    != runmat_execution::identity::ArtifactId::derive(&[program
                        .artifact
                        .id
                        .0
                        .bytes()])
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
                for output in &commit.outputs {
                    output.validate(ValueLimits::default()).map_err(invalid)?;
                }
                if super::super::value_transfer::collect_references(&commit.outputs)
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
    /// Capture metadata under admission-compatible locks; no object IO occurs under these locks.
    pub fn capture_checkpoint(
        &self,
        generation: u64,
    ) -> NativeExecutionResult<RemotePoolCheckpoint> {
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
                let program = programs
                    .get(id)
                    .ok_or_else(|| invalid("admitted program missing during checkpoint"))?;
                inventories.validate(*id, program)?;
                Ok((
                    *id,
                    TaskBinding {
                        program: program.clone(),
                        inventory: inventories.get(*id),
                    },
                ))
            })
            .collect::<NativeExecutionResult<BTreeMap<_, _>>>()?;
        let mut execution_inputs = self.execution_objects.checkpoint_inputs();
        execution_inputs.sort_by_key(|reference| reference.id);
        let mut value_inputs = self.values.checkpoint_references();
        value_inputs.sort_by_key(|reference| reference.id);
        let checkpoint = RemotePoolCheckpoint {
            schema_version: 1,
            scope_id: self.scope_id,
            pool_id: self.pool_id,
            value_scope: self.value_scope.clone(),
            bundle_digest: self.bundle_digest,
            scheduler: DriverCheckpoint::new(generation, snapshot),
            tasks,
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
            || checkpoint.scheduler.snapshot.pools[&checkpoint.pool_id].spec != authority.pool
        {
            return Err(invalid(
                "checkpoint differs from replacement allocation authority",
            ));
        }
        let driver = checkpoint
            .scheduler
            .clone()
            .restore_at_fence(authority.driver_fence)?;
        let mut pool = Self::new_with_value_scope(
            authority.scope_id,
            authority.pool,
            authority.driver_fence,
            bundle,
            authority.value_scope,
        )?;
        let restored = Arc::get_mut(&mut pool).expect("new pool is exclusively owned");
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
                            .result_objects
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
                    .commit_results(&commit.result_objects)?;
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
            programs.insert(id, binding.program);
        }
        *restored.driver.get_mut().expect("new driver") = driver;
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
