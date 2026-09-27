use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use runmat_execution::identity::ValueId;
use runmat_execution::value::{ValueLimits, ValuePayload, ValueRef};
use runmat_execution::Digest;
use runmat_execution_artifact::cache::{CacheExport, CacheImport, FilesystemObjectStore};
use runmat_execution_artifact::{LogicalObject, ObjectNamespace};
use tokio::sync::Mutex as AsyncMutex;

use super::object_transfer::{references_share_content_identity, MAX_REMOTE_OBJECTS};
use super::RemoteWorkerChannel;
use crate::{NativeExecutionError, NativeExecutionResult};

#[derive(Default)]
struct ObjectState {
    objects: HashMap<ValueId, ValueRef>,
    registered_inputs: HashSet<ValueId>,
    committed_results: HashSet<ValueId>,
    pending_results: HashMap<ValueId, (ValueRef, usize)>,
}

/// Owns downloaded results until scheduler admission protects them or rejects them.
/// Cancellation of the receiving future releases the same reservation as normal drop.
#[must_use = "retain the result lease until scheduler admission completes"]
pub(super) struct PendingResultLease<'a> {
    catalog: &'a RemoteObjectCatalog,
    references: Vec<ValueRef>,
}

impl Drop for PendingResultLease<'_> {
    fn drop(&mut self) {
        let mut state = self
            .catalog
            .state
            .lock()
            .expect("remote object catalog poisoned");
        for reference in &self.references {
            if let Some((reserved, count)) = state.pending_results.get_mut(&reference.id) {
                if references_share_content_identity(reserved, reference) {
                    *count -= 1;
                    if *count == 0 {
                        state.pending_results.remove(&reference.id);
                    }
                }
            }
            discard_unowned(&mut state, reference);
        }
    }
}

pub(super) struct RemoteObjectCatalog {
    state: Mutex<ObjectState>,
    transferred: AsyncMutex<HashSet<(super::pool::WorkerRouteIdentity, ValueId)>>,
    store: FilesystemObjectStore,
    _directory: tempfile::TempDir,
}

impl RemoteObjectCatalog {
    pub(super) fn new() -> NativeExecutionResult<Self> {
        let directory = tempfile::Builder::new()
            .prefix("runmat-remote-pool-objects-")
            .tempdir()
            .map_err(protocol)?;
        let store = FilesystemObjectStore::open(
            directory.path().join("objects"),
            super::object_transfer::MAX_REMOTE_OBJECT_BYTES,
        )
        .map_err(protocol)?;
        Ok(Self {
            state: Mutex::new(ObjectState::default()),
            transferred: AsyncMutex::new(HashSet::new()),
            store,
            _directory: directory,
        })
    }
    pub(super) fn register(
        &self,
        reference: ValueRef,
        encoded: Arc<[u8]>,
        authorization_scope: &str,
    ) -> NativeExecutionResult<()> {
        validate_object(&reference, &encoded, authorization_scope)?;
        let mut state = self.state.lock().expect("remote object catalog poisoned");
        let mut admitted = admitted_objects(&state)?;
        validate_extension(&admitted, std::iter::once(&reference))?;
        insert(&mut admitted, reference.clone())?;
        let value_id = reference.id;
        store(&self.store, &reference, encoded.as_ref())?;
        insert(&mut state.objects, reference)?;
        state.registered_inputs.insert(value_id);
        Ok(())
    }

    pub(super) fn get(&self, reference: &ValueRef) -> NativeExecutionResult<Option<Arc<[u8]>>> {
        let state = self.state.lock().expect("remote object catalog poisoned");
        match state.objects.get(&reference.id) {
            Some(stored) if references_share_content_identity(stored, reference) => self
                .store
                .read_verified(reference.logical_digest)
                .map(|bytes| bytes.map(Arc::<[u8]>::from))
                .map_err(protocol),
            Some(_) => Err(protocol(
                "remote execution object reference differs from stored content",
            )),
            None => Ok(None),
        }
    }

    pub(super) fn get_by_digest(&self, digest: Digest) -> NativeExecutionResult<Option<Arc<[u8]>>> {
        let state = self.state.lock().expect("remote object catalog poisoned");
        let reference = state
            .objects
            .values()
            .find(|reference| reference.logical_digest == digest);
        match reference {
            Some(_) => self
                .store
                .read_verified(digest)
                .map(|bytes| bytes.map(Arc::<[u8]>::from))
                .map_err(protocol),
            None => Ok(None),
        }
    }

    pub(super) fn contains(&self, reference: &ValueRef) -> NativeExecutionResult<bool> {
        let state = self.state.lock().expect("remote object catalog poisoned");
        match state.objects.get(&reference.id) {
            Some(stored) if references_share_content_identity(stored, reference) => Ok(true),
            Some(_) => Err(protocol(
                "remote execution object reference differs from stored content",
            )),
            None => Ok(false),
        }
    }

    pub(super) async fn transfer(
        &self,
        channel: &dyn RemoteWorkerChannel,
        route: super::pool::WorkerRouteIdentity,
        inventory: Option<&[ValueRef]>,
    ) -> NativeExecutionResult<()> {
        let references = match inventory {
            Some(references) => {
                for reference in references {
                    if !self.contains(reference)? {
                        return Err(protocol("task input object is absent"));
                    }
                }
                references.to_vec()
            }
            None => self
                .state
                .lock()
                .expect("remote object catalog poisoned")
                .objects
                .values()
                .cloned()
                .collect::<Vec<_>>(),
        };
        for reference in references {
            let key = (route, reference.id);
            if self.transferred.lock().await.contains(&key) {
                continue;
            }
            let encoded = self
                .store
                .read_verified(reference.logical_digest)
                .map_err(protocol)?
                .ok_or_else(|| protocol("registered remote input object is absent"))?;
            let receipt = channel.transfer_object(reference.clone(), &encoded).await?;
            if receipt.value_id != reference.id
                || receipt.next_offset != reference.encoded_length
                || !receipt.complete
            {
                return Err(protocol(
                    "remote worker did not acknowledge the complete execution object",
                ));
            }
            self.transferred.lock().await.insert(key);
        }
        Ok(())
    }

    pub(super) async fn receive_results(
        &self,
        channel: &dyn RemoteWorkerChannel,
        references: &[ValueRef],
        authorization_scope: &str,
    ) -> NativeExecutionResult<PendingResultLease<'_>> {
        let lease = self.reserve_results(references)?;
        for reference in &lease.references {
            let encoded = channel.download_object(reference.clone()).await?;
            validate_object(reference, &encoded, authorization_scope)?;
            store(&self.store, reference, &encoded)?;
            let mut state = self.state.lock().expect("remote object catalog poisoned");
            insert(&mut state.objects, reference.clone())?;
        }
        Ok(lease)
    }

    fn reserve_results(
        &self,
        references: &[ValueRef],
    ) -> NativeExecutionResult<PendingResultLease<'_>> {
        let mut state = self.state.lock().expect("remote object catalog poisoned");
        // Stage the entire admission first: pending downloads consume inventory
        // capacity and establish identity even before bytes reach the catalog.
        let mut admitted = admitted_objects(&state)?;
        validate_extension(&admitted, references.iter())?;
        let mut seen = HashSet::new();
        let mut unique = Vec::new();
        for reference in references {
            insert(&mut admitted, reference.clone())?;
            if seen.insert(reference.id) {
                unique.push(reference.clone());
            }
        }
        for reference in &unique {
            if state
                .pending_results
                .get(&reference.id)
                .is_some_and(|(_, count)| *count == usize::MAX)
            {
                return Err(protocol("remote pending-result ownership overflow"));
            }
        }
        let references = unique;
        for reference in &references {
            let (_, count) = state
                .pending_results
                .entry(reference.id)
                .or_insert_with(|| (reference.clone(), 0));
            *count += 1;
        }
        Ok(PendingResultLease {
            catalog: self,
            references,
        })
    }

    pub(super) fn commit_results(&self, references: &[ValueRef]) -> NativeExecutionResult<()> {
        let mut state = self.state.lock().expect("remote object catalog poisoned");
        for reference in references {
            match state.objects.get(&reference.id) {
                Some(stored) if references_share_content_identity(stored, reference) => {}
                Some(_) => {
                    return Err(protocol(
                        "committed remote result differs from downloaded content",
                    ))
                }
                None => return Err(protocol("committed remote result was not downloaded")),
            }
        }
        state
            .committed_results
            .extend(references.iter().map(|reference| reference.id));
        Ok(())
    }

    pub(super) fn discard_results(&self, references: &[ValueRef]) {
        let mut state = self.state.lock().expect("remote object catalog poisoned");
        for reference in references {
            discard_unowned(&mut state, reference);
        }
    }
}

fn admitted_objects(state: &ObjectState) -> NativeExecutionResult<HashMap<ValueId, ValueRef>> {
    let mut admitted = state.objects.clone();
    for (reference, _) in state.pending_results.values() {
        insert(&mut admitted, reference.clone())?;
    }
    Ok(admitted)
}

fn discard_unowned(state: &mut ObjectState, reference: &ValueRef) {
    if !state.registered_inputs.contains(&reference.id)
        && !state.committed_results.contains(&reference.id)
        && !state.pending_results.contains_key(&reference.id)
        && state
            .objects
            .get(&reference.id)
            .is_some_and(|stored| references_share_content_identity(stored, reference))
    {
        state.objects.remove(&reference.id);
    }
}

fn validate_object(
    reference: &ValueRef,
    encoded: &[u8],
    authorization_scope: &str,
) -> NativeExecutionResult<()> {
    ValuePayload::Object(Box::new(reference.clone()))
        .validate(ValueLimits::default())
        .map_err(protocol)?;
    if reference.authorization_scope != authorization_scope
        || reference.encoded_length == 0
        || reference.encoded_length != encoded.len() as u64
        || reference.logical_digest != Digest::sha256(encoded)
    {
        return Err(protocol(
            "remote execution object differs from its identity or authority",
        ));
    }
    Ok(())
}

fn insert(
    objects: &mut HashMap<ValueId, ValueRef>,
    reference: ValueRef,
) -> NativeExecutionResult<()> {
    if let Some(stored) = objects.get(&reference.id) {
        if references_share_content_identity(stored, &reference) {
            return Ok(());
        }
        return Err(protocol(
            "remote execution object id was reused for different content",
        ));
    }
    objects.insert(reference.id, reference);
    Ok(())
}

fn validate_extension<'a>(
    objects: &HashMap<ValueId, ValueRef>,
    additions: impl Iterator<Item = &'a ValueRef>,
) -> NativeExecutionResult<()> {
    let mut count = objects.len();
    let mut total = objects
        .values()
        .try_fold(0_u64, |total, reference| {
            total.checked_add(reference.encoded_length)
        })
        .ok_or_else(|| protocol("remote execution object byte total overflowed"))?;
    let mut new_ids = HashSet::new();
    for reference in additions {
        if objects.contains_key(&reference.id) || !new_ids.insert(reference.id) {
            continue;
        }
        count = count.saturating_add(1);
        total = total
            .checked_add(reference.encoded_length)
            .ok_or_else(|| protocol("remote execution object byte total overflowed"))?;
    }
    if count > MAX_REMOTE_OBJECTS
        || total > runmat_execution_artifact::MAX_PROGRAM_EXECUTION_OBJECT_TOTAL_BYTES
    {
        return Err(protocol(
            "remote execution object catalog exceeds its hard bounds",
        ));
    }
    Ok(())
}

fn store(
    store: &FilesystemObjectStore,
    reference: &ValueRef,
    bytes: &[u8],
) -> NativeExecutionResult<()> {
    let namespace = match reference.kind {
        runmat_execution::value::ValueRefKind::DriverObject => ObjectNamespace::InputValue,
        runmat_execution::value::ValueRefKind::ResultObject => ObjectNamespace::ResultValue,
        _ => return Err(protocol("remote execution object has a non-portable kind")),
    };
    let object = LogicalObject::new(
        namespace,
        reference.value_schema.clone(),
        reference.media_type.clone(),
        bytes.to_vec(),
    )
    .map_err(protocol)?;
    if object.descriptor.digest != reference.logical_digest
        || object.descriptor.encoded_length != reference.encoded_length
    {
        return Err(protocol(
            "remote execution object differs from its logical descriptor",
        ));
    }
    let mut store = store.clone();
    store.write_verified(&object).map_err(protocol)
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

#[cfg(test)]
mod tests {
    use runmat_execution::value::ValueRefKind;

    use super::*;

    #[test]
    fn pending_owners_release_independently_and_future_drop_cleans_uncommitted_objects() {
        let catalog = RemoteObjectCatalog::new().unwrap();
        let reference = reference("run", b"pending", b"pending");
        let first = catalog
            .reserve_results(&[reference.clone(), reference.clone()])
            .unwrap();
        let second = catalog
            .reserve_results(std::slice::from_ref(&reference))
            .unwrap();
        store(&catalog.store, &reference, b"pending").unwrap();
        insert(
            &mut catalog.state.lock().unwrap().objects,
            reference.clone(),
        )
        .unwrap();
        assert_eq!(
            catalog.state.lock().unwrap().pending_results[&reference.id].1,
            2
        );
        catalog.discard_results(std::slice::from_ref(&reference));
        drop(first);
        assert!(catalog.contains(&reference).unwrap());
        // Dropping a suspended/abandoned task drops its owned lease without an
        // async cleanup operation or a scheduler callback.
        let abandoned = async move {
            let _lease = second;
            std::future::pending::<()>().await;
        };
        drop(abandoned);
        assert!(catalog.state.lock().unwrap().pending_results.is_empty());
        assert!(!catalog.contains(&reference).unwrap());
    }

    #[test]
    fn pending_identity_and_capacity_are_reserved_atomically_before_download() {
        let catalog = RemoteObjectCatalog::new().unwrap();
        let reference = reference("run", b"reserved", b"reserved");
        let lease = catalog
            .reserve_results(std::slice::from_ref(&reference))
            .unwrap();
        let mut foreign = reference.clone();
        foreign.logical_digest = Digest::sha256(b"foreign");
        let unrelated = self::reference("run", b"unrelated", b"unrelated");
        assert!(catalog
            .reserve_results(&[unrelated.clone(), foreign])
            .is_err());
        assert_eq!(catalog.state.lock().unwrap().pending_results.len(), 1);
        assert_eq!(
            catalog.state.lock().unwrap().pending_results[&reference.id].1,
            1
        );
        drop(lease);
        let mut oversized = reference.clone();
        oversized.encoded_length =
            runmat_execution_artifact::MAX_PROGRAM_EXECUTION_OBJECT_TOTAL_BYTES;
        let full = catalog.reserve_results(&[oversized]).unwrap();
        assert!(catalog
            .reserve_results(std::slice::from_ref(&unrelated))
            .is_err());
        assert!(catalog
            .register(unrelated, Arc::from(&b"unrelated"[..]), "run")
            .is_err());
        assert_eq!(catalog.state.lock().unwrap().pending_results.len(), 1);
        drop(full);
        assert!(catalog.state.lock().unwrap().pending_results.is_empty());
    }

    #[test]
    fn stale_results_are_discarded_without_evicting_inputs_or_commits() {
        let catalog = RemoteObjectCatalog::new().unwrap();
        let input_bytes = Arc::<[u8]>::from(&b"registered input"[..]);
        let input = reference("run", input_bytes.as_ref(), b"input");
        catalog
            .register(input.clone(), Arc::clone(&input_bytes), "run")
            .unwrap();
        assert!(catalog.contains(&input).unwrap());
        let mut driver_alias = input.clone();
        driver_alias.kind = ValueRefKind::DriverObject;
        catalog
            .register(driver_alias.clone(), Arc::clone(&input_bytes), "run")
            .unwrap();
        assert!(catalog.contains(&driver_alias).unwrap());
        let mut conflicting_input = input.clone();
        conflicting_input.encoded_length += 1;
        assert!(catalog.contains(&conflicting_input).is_err());

        let stale_bytes = Arc::<[u8]>::from(&b"stale result"[..]);
        let stale = reference("run", stale_bytes.as_ref(), b"stale");
        let committed_bytes = Arc::<[u8]>::from(&b"committed result"[..]);
        let committed = reference("run", committed_bytes.as_ref(), b"committed");
        store(&catalog.store, &stale, stale_bytes.as_ref()).unwrap();
        store(&catalog.store, &committed, committed_bytes.as_ref()).unwrap();
        {
            let mut state = catalog.state.lock().unwrap();
            insert(&mut state.objects, stale.clone()).unwrap();
            insert(&mut state.objects, committed.clone()).unwrap();
        }
        catalog
            .commit_results(std::slice::from_ref(&committed))
            .unwrap();
        catalog.discard_results(&[stale.clone(), input.clone(), committed.clone()]);

        assert!(catalog.get(&stale).unwrap().is_none());
        assert!(!catalog.contains(&stale).unwrap());
        assert!(catalog.get(&input).unwrap().is_some());
        assert_eq!(
            catalog.get(&committed).unwrap().unwrap().as_ref(),
            committed_bytes.as_ref()
        );
    }

    fn reference(scope: &str, bytes: &[u8], identity: &[u8]) -> ValueRef {
        ValueRef {
            schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_V1,
            id: ValueId::derive(&[b"remote-object-test", identity]),
            logical_digest: Digest::sha256(bytes),
            encoded_length: bytes.len() as u64,
            media_type: "application/vnd.runmat.test-object".into(),
            value_schema: "runmat.test-object.v1".into(),
            encryption_context: Digest::sha256(b"remote-object-test-context"),
            kind: ValueRefKind::ResultObject,
            authorization_scope: scope.into(),
            resident_fence: None,
        }
    }
}
