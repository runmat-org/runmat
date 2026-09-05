use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use runmat_execution::identity::{ValueId, WorkerId};
use runmat_execution::value::{ValueLimits, ValuePayload, ValueRef};
use runmat_execution::Digest;
use runmat_execution_artifact::cache::{CacheExport, CacheImport, FilesystemObjectStore};
use runmat_execution_artifact::{LogicalObject, ObjectNamespace};
use tokio::sync::Mutex as AsyncMutex;

use super::object_transfer::MAX_REMOTE_OBJECTS;
use super::RemoteWorkerChannel;
use crate::{NativeExecutionError, NativeExecutionResult};

#[derive(Default)]
struct ObjectState {
    objects: HashMap<ValueId, ValueRef>,
    registered_inputs: HashSet<ValueId>,
    committed_results: HashSet<ValueId>,
}

pub(super) struct RemoteObjectCatalog {
    state: Mutex<ObjectState>,
    transferred: AsyncMutex<HashSet<(WorkerId, ValueId)>>,
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
        validate_extension(&state.objects, std::iter::once(&reference))?;
        let value_id = reference.id;
        store(&self.store, &reference, encoded.as_ref())?;
        insert(&mut state.objects, reference)?;
        state.registered_inputs.insert(value_id);
        Ok(())
    }

    pub(super) fn get(&self, reference: &ValueRef) -> NativeExecutionResult<Option<Arc<[u8]>>> {
        let state = self.state.lock().expect("remote object catalog poisoned");
        match state.objects.get(&reference.id) {
            Some(stored) if stored == reference => self
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
            Some(stored) if stored == reference => Ok(true),
            Some(_) => Err(protocol(
                "remote execution object reference differs from stored content",
            )),
            None => Ok(false),
        }
    }

    pub(super) async fn transfer_all(
        &self,
        channel: &dyn RemoteWorkerChannel,
        worker_id: WorkerId,
    ) -> NativeExecutionResult<()> {
        let references = self
            .state
            .lock()
            .expect("remote object catalog poisoned")
            .objects
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for reference in references {
            let key = (worker_id, reference.id);
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
    ) -> NativeExecutionResult<()> {
        {
            let state = self.state.lock().expect("remote object catalog poisoned");
            for reference in references {
                if let Some(stored) = state.objects.get(&reference.id) {
                    if stored != reference {
                        return Err(protocol(
                            "remote result object collides with registered content",
                        ));
                    }
                }
            }
            validate_extension(&state.objects, references.iter())?;
        }
        let mut downloaded = Vec::with_capacity(references.len());
        for reference in references {
            let encoded = channel.download_object(reference.clone()).await?;
            validate_object(reference, &encoded, authorization_scope)?;
            store(&self.store, reference, &encoded)?;
            downloaded.push(reference.clone());
        }
        let mut state = self.state.lock().expect("remote object catalog poisoned");
        for reference in downloaded {
            insert(&mut state.objects, reference)?;
        }
        Ok(())
    }

    pub(super) fn commit_results(&self, references: &[ValueRef]) -> NativeExecutionResult<()> {
        let mut state = self.state.lock().expect("remote object catalog poisoned");
        for reference in references {
            match state.objects.get(&reference.id) {
                Some(stored) if stored == reference => {}
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
            if !state.registered_inputs.contains(&reference.id)
                && !state.committed_results.contains(&reference.id)
                && state
                    .objects
                    .get(&reference.id)
                    .is_some_and(|stored| stored == reference)
            {
                state.objects.remove(&reference.id);
            }
        }
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
        if stored == &reference {
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
    fn stale_results_are_discarded_without_evicting_inputs_or_commits() {
        let catalog = RemoteObjectCatalog::new().unwrap();
        let input_bytes = Arc::<[u8]>::from(&b"registered input"[..]);
        let input = reference("run", input_bytes.as_ref(), b"input");
        catalog
            .register(input.clone(), Arc::clone(&input_bytes), "run")
            .unwrap();
        assert!(catalog.contains(&input).unwrap());
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
