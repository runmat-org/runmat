use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use runmat_execution::identity::{ValueId, WorkerId};
use runmat_execution::value::{ValueLimits, ValuePayload, ValueRef};
use runmat_execution::Digest;
use tokio::sync::Mutex as AsyncMutex;

use super::object_transfer::{MAX_REMOTE_OBJECTS, MAX_REMOTE_OBJECT_TOTAL_BYTES};
use super::RemoteWorkerChannel;
use crate::{NativeExecutionError, NativeExecutionResult};

type StoredObject = (ValueRef, Arc<[u8]>);

#[derive(Default)]
struct ObjectState {
    objects: HashMap<ValueId, StoredObject>,
    registered_inputs: HashSet<ValueId>,
    committed_results: HashSet<ValueId>,
}

#[derive(Default)]
pub(super) struct RemoteObjectCatalog {
    state: Mutex<ObjectState>,
    transferred: AsyncMutex<HashSet<(WorkerId, ValueId)>>,
}

impl RemoteObjectCatalog {
    pub(super) fn register(
        &self,
        reference: ValueRef,
        encoded: Arc<[u8]>,
        authorization_scope: &str,
    ) -> NativeExecutionResult<()> {
        validate_object(&reference, &encoded, authorization_scope)?;
        let mut state = self.state.lock().expect("remote object catalog poisoned");
        validate_extension(
            &state.objects,
            std::iter::once((&reference, encoded.len() as u64)),
        )?;
        let value_id = reference.id;
        insert(&mut state.objects, reference, encoded)?;
        state.registered_inputs.insert(value_id);
        Ok(())
    }

    pub(super) fn get(&self, reference: &ValueRef) -> NativeExecutionResult<Option<Arc<[u8]>>> {
        let state = self.state.lock().expect("remote object catalog poisoned");
        match state.objects.get(&reference.id) {
            Some((stored, bytes)) if stored == reference => Ok(Some(Arc::clone(bytes))),
            Some(_) => Err(protocol(
                "remote execution object reference differs from stored content",
            )),
            None => Ok(None),
        }
    }

    pub(super) async fn transfer_all(
        &self,
        channel: &dyn RemoteWorkerChannel,
        worker_id: WorkerId,
    ) -> NativeExecutionResult<()> {
        let objects = self
            .state
            .lock()
            .expect("remote object catalog poisoned")
            .objects
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for (reference, encoded) in objects {
            let key = (worker_id, reference.id);
            if self.transferred.lock().await.contains(&key) {
                continue;
            }
            let receipt = channel
                .transfer_object(reference.clone(), encoded.as_ref())
                .await?;
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
        let mut downloaded = Vec::with_capacity(references.len());
        for reference in references {
            let encoded = channel.download_object(reference.clone()).await?;
            validate_object(reference, &encoded, authorization_scope)?;
            downloaded.push((reference.clone(), Arc::<[u8]>::from(encoded)));
        }
        let mut state = self.state.lock().expect("remote object catalog poisoned");
        for (reference, encoded) in &downloaded {
            if let Some((stored, bytes)) = state.objects.get(&reference.id) {
                if stored != reference || bytes.as_ref() != encoded.as_ref() {
                    return Err(protocol(
                        "remote result object collides with registered content",
                    ));
                }
            }
        }
        validate_extension(
            &state.objects,
            downloaded
                .iter()
                .map(|(reference, encoded)| (reference, encoded.len() as u64)),
        )?;
        for (reference, encoded) in downloaded {
            insert(&mut state.objects, reference, encoded)?;
        }
        Ok(())
    }

    pub(super) fn commit_results(&self, references: &[ValueRef]) -> NativeExecutionResult<()> {
        let mut state = self.state.lock().expect("remote object catalog poisoned");
        for reference in references {
            match state.objects.get(&reference.id) {
                Some((stored, _)) if stored == reference => {}
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
                    .is_some_and(|(stored, _)| stored == reference)
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
    objects: &mut HashMap<ValueId, StoredObject>,
    reference: ValueRef,
    encoded: Arc<[u8]>,
) -> NativeExecutionResult<()> {
    if let Some((stored, bytes)) = objects.get(&reference.id) {
        if stored == &reference && bytes.as_ref() == encoded.as_ref() {
            return Ok(());
        }
        return Err(protocol(
            "remote execution object id was reused for different content",
        ));
    }
    objects.insert(reference.id, (reference, encoded));
    Ok(())
}

fn validate_extension<'a>(
    objects: &HashMap<ValueId, StoredObject>,
    additions: impl Iterator<Item = (&'a ValueRef, u64)>,
) -> NativeExecutionResult<()> {
    let mut count = objects.len();
    let mut total = objects
        .values()
        .try_fold(0_u64, |total, (_, bytes)| {
            total.checked_add(bytes.len() as u64)
        })
        .ok_or_else(|| protocol("remote execution object byte total overflowed"))?;
    let mut new_ids = HashSet::new();
    for (reference, length) in additions {
        if objects.contains_key(&reference.id) || !new_ids.insert(reference.id) {
            continue;
        }
        count = count.saturating_add(1);
        total = total
            .checked_add(length)
            .ok_or_else(|| protocol("remote execution object byte total overflowed"))?;
    }
    if count > MAX_REMOTE_OBJECTS || total > MAX_REMOTE_OBJECT_TOTAL_BYTES {
        return Err(protocol(
            "remote execution object catalog exceeds its hard bounds",
        ));
    }
    Ok(())
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
        let catalog = RemoteObjectCatalog::default();
        let input_bytes = Arc::<[u8]>::from(&b"registered input"[..]);
        let input = reference("run", input_bytes.as_ref(), b"input");
        catalog
            .register(input.clone(), Arc::clone(&input_bytes), "run")
            .unwrap();

        let stale_bytes = Arc::<[u8]>::from(&b"stale result"[..]);
        let stale = reference("run", stale_bytes.as_ref(), b"stale");
        let committed_bytes = Arc::<[u8]>::from(&b"committed result"[..]);
        let committed = reference("run", committed_bytes.as_ref(), b"committed");
        {
            let mut state = catalog.state.lock().unwrap();
            insert(&mut state.objects, stale.clone(), stale_bytes).unwrap();
            insert(
                &mut state.objects,
                committed.clone(),
                Arc::clone(&committed_bytes),
            )
            .unwrap();
        }
        catalog
            .commit_results(std::slice::from_ref(&committed))
            .unwrap();
        catalog.discard_results(&[stale.clone(), input.clone(), committed.clone()]);

        assert!(catalog.get(&stale).unwrap().is_none());
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
