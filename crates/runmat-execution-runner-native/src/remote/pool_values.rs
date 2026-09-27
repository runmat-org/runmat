use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use runmat_execution::identity::ValueId;
use runmat_execution::value::{ValuePayload, ValueRef};
use tokio::sync::Mutex as AsyncMutex;

use super::pool_objects::RemoteObjectCatalog;
use super::RemoteWorkerChannel;
use crate::{NativeExecutionError, NativeExecutionResult};

type StoredValue = (ValueRef, Arc<[u8]>);

#[derive(Default)]
pub(super) struct RemoteValueCatalog {
    objects: Mutex<HashMap<ValueId, StoredValue>>,
    transferred: AsyncMutex<HashSet<(super::pool::WorkerRouteIdentity, ValueId)>>,
}

impl RemoteValueCatalog {
    pub(super) fn checkpoint_references(&self) -> Vec<ValueRef> {
        self.objects
            .lock()
            .expect("remote value catalog poisoned")
            .values()
            .map(|(reference, _)| reference.clone())
            .collect()
    }

    pub(super) fn encoded(&self, reference: &ValueRef) -> NativeExecutionResult<Option<Arc<[u8]>>> {
        let objects = self.objects.lock().expect("remote value catalog poisoned");
        match objects.get(&reference.id) {
            Some((stored, bytes)) if stored == reference => Ok(Some(Arc::clone(bytes))),
            Some(_) => Err(protocol(
                "remote value reference differs from stored content",
            )),
            None => Ok(None),
        }
    }
    pub(super) fn register(
        &self,
        reference: ValueRef,
        encoded: Arc<[u8]>,
        authorization_scope: &str,
    ) -> NativeExecutionResult<()> {
        super::value_transfer::decode_value(&reference, &encoded, authorization_scope)?;
        let mut objects = self.objects.lock().expect("remote value catalog poisoned");
        if let Some((stored, bytes)) = objects.get(&reference.id) {
            if stored == &reference && bytes.as_ref() == encoded.as_ref() {
                return Ok(());
            }
            return Err(protocol(
                "remote value object id was reused for different content",
            ));
        }
        objects.insert(reference.id, (reference, encoded));
        Ok(())
    }

    pub(super) async fn transfer(
        &self,
        channel: &dyn RemoteWorkerChannel,
        route: super::pool::WorkerRouteIdentity,
        values: &[ValuePayload],
        execution_objects: &RemoteObjectCatalog,
    ) -> NativeExecutionResult<()> {
        let references = super::value_transfer::collect_references(values);
        for reference in references {
            if execution_objects.contains(&reference)? {
                continue;
            }
            let key = (route, reference.id);
            if self.transferred.lock().await.contains(&key) {
                continue;
            }
            let (stored, encoded) = self
                .objects
                .lock()
                .expect("remote value catalog poisoned")
                .get(&reference.id)
                .cloned()
                .ok_or_else(|| {
                    protocol(format!(
                        "remote value object {} is not registered",
                        reference.id
                    ))
                })?;
            if stored != reference {
                return Err(protocol(
                    "remote value reference differs from its registered object",
                ));
            }
            let receipt = channel.transfer_value(reference.clone(), &encoded).await?;
            if receipt.value_id != reference.id || receipt.encoded_bytes != encoded.len() as u64 {
                return Err(protocol(
                    "remote worker acknowledged a different value object",
                ));
            }
            self.transferred.lock().await.insert(key);
        }
        Ok(())
    }

    pub(super) fn resolve(
        &self,
        reference: &ValueRef,
        authorization_scope: &str,
    ) -> NativeExecutionResult<ValuePayload> {
        let (stored, encoded) = self
            .objects
            .lock()
            .expect("remote value catalog poisoned")
            .get(&reference.id)
            .cloned()
            .ok_or_else(|| protocol("remote value object is not registered"))?;
        if &stored != reference {
            return Err(protocol(
                "remote value reference differs from its registered object",
            ));
        }
        super::value_transfer::decode_value(&stored, &encoded, authorization_scope)
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
