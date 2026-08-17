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
type ObjectMap = HashMap<ValueId, StoredObject>;

#[derive(Default)]
pub(super) struct RemoteObjectCatalog {
    objects: Mutex<ObjectMap>,
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
        let mut objects = self.objects.lock().expect("remote object catalog poisoned");
        validate_extension(
            &objects,
            std::iter::once((&reference, encoded.len() as u64)),
        )?;
        insert(&mut objects, reference, encoded)
    }

    pub(super) fn get(&self, reference: &ValueRef) -> NativeExecutionResult<Option<Arc<[u8]>>> {
        let objects = self.objects.lock().expect("remote object catalog poisoned");
        match objects.get(&reference.id) {
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
            .objects
            .lock()
            .expect("remote object catalog poisoned")
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
        let mut objects = self.objects.lock().expect("remote object catalog poisoned");
        for (reference, encoded) in &downloaded {
            if let Some((stored, bytes)) = objects.get(&reference.id) {
                if stored != reference || bytes.as_ref() != encoded.as_ref() {
                    return Err(protocol(
                        "remote result object collides with registered content",
                    ));
                }
            }
        }
        validate_extension(
            &objects,
            downloaded
                .iter()
                .map(|(reference, encoded)| (reference, encoded.len() as u64)),
        )?;
        for (reference, encoded) in downloaded {
            insert(&mut objects, reference, encoded)?;
        }
        Ok(())
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
    objects: &mut ObjectMap,
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
    objects: &ObjectMap,
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
