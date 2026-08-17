use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use runmat_execution::identity::{ValueId, WorkerId};
use runmat_execution::value::{ValuePayload, ValueRef};
use tokio::sync::Mutex as AsyncMutex;

use super::RemoteWorkerChannel;
use crate::{NativeExecutionError, NativeExecutionResult};

type StoredValue = (ValueRef, Arc<[u8]>);

#[derive(Default)]
pub(super) struct RemoteValueCatalog {
    objects: Mutex<HashMap<ValueId, StoredValue>>,
    transferred: AsyncMutex<HashSet<(WorkerId, ValueId)>>,
}

impl RemoteValueCatalog {
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
        worker_id: WorkerId,
        values: &[ValuePayload],
    ) -> NativeExecutionResult<()> {
        let references = super::value_transfer::collect_references(values);
        for reference in references {
            let key = (worker_id, reference.id);
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
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
