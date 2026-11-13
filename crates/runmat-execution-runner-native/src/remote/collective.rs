use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use runmat_execution::identity::AttemptId;
use runmat_execution::{
    CollectiveRequest, CollectiveResponse, CollectiveSequence, SpmdTaskContext,
};
use runmat_execution_transport_native::frame::{EncryptedFrameSession, FrameLimits};
use runmat_runtime::context::{RuntimeCollectiveService, RuntimeServiceFuture};
use runmat_runtime::RuntimeError;
use runmat_types::CollectiveId;
use tokio::sync::{oneshot, Mutex as AsyncMutex};

use super::protocol::{RemoteWorkerOutcome, RemoteWorkerReply, REMOTE_WORKER_PROTOCOL_VERSION};
use super::route::RemoteFrameRoute;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CompletionKey {
    id: CollectiveId,
    sequence: CollectiveSequence,
}

#[derive(Default)]
struct CollectiveCompletionRegistry(
    Mutex<BTreeMap<CompletionKey, oneshot::Sender<crate::protocol::CollectiveProcessResult>>>,
);

impl CollectiveCompletionRegistry {
    fn register(
        &self,
        key: CompletionKey,
        sender: oneshot::Sender<crate::protocol::CollectiveProcessResult>,
    ) -> Result<(), String> {
        if self
            .0
            .lock()
            .map_err(|_| "remote collective completion registry is poisoned".to_string())?
            .insert(key, sender)
            .is_some()
        {
            return Err("remote collective request reused an active identity".into());
        }
        Ok(())
    }

    fn remove(&self, key: CompletionKey) {
        self.0
            .lock()
            .expect("remote collective completion registry poisoned")
            .remove(&key);
    }

    fn complete(
        &self,
        key: CompletionKey,
        result: crate::protocol::CollectiveProcessResult,
    ) -> Result<(), String> {
        let sender = self
            .0
            .lock()
            .map_err(|_| "remote collective completion registry is poisoned".to_string())?
            .remove(&key)
            .ok_or_else(|| {
                "remote collective completion identity is unknown or stale".to_string()
            })?;
        sender
            .send(result)
            .map_err(|_| "remote collective caller stopped before completion".to_string())
    }
}

impl CompletionKey {
    fn for_request(request: &CollectiveRequest) -> Self {
        Self {
            id: request.id,
            sequence: request.sequence,
        }
    }
}

pub(super) struct RemoteCollectiveChannel {
    attempt_id: AttemptId,
    correlation_id: String,
    context: SpmdTaskContext,
    connection: Arc<dyn RemoteFrameRoute>,
    sender: Arc<AsyncMutex<EncryptedFrameSession>>,
    limits: FrameLimits,
    pending: CollectiveCompletionRegistry,
}

impl RemoteCollectiveChannel {
    pub(super) fn new(
        attempt_id: AttemptId,
        correlation_id: String,
        context: SpmdTaskContext,
        connection: Arc<dyn RemoteFrameRoute>,
        sender: Arc<AsyncMutex<EncryptedFrameSession>>,
        limits: FrameLimits,
    ) -> Arc<Self> {
        Arc::new(Self {
            attempt_id,
            correlation_id,
            context,
            connection,
            sender,
            limits,
            pending: CollectiveCompletionRegistry::default(),
        })
    }

    async fn submit(
        self: Arc<Self>,
        request: CollectiveRequest,
    ) -> Result<CollectiveResponse, RuntimeError> {
        request.validate().map_err(transport_error)?;
        if request.context != self.context {
            return Err(transport_error(
                "remote collective request differs from its admitted rank context",
            ));
        }
        let key = CompletionKey::for_request(&request);
        let (sender, receiver) = oneshot::channel();
        self.pending
            .register(key, sender)
            .map_err(transport_error)?;
        let reply = RemoteWorkerReply {
            schema_version: REMOTE_WORKER_PROTOCOL_VERSION,
            correlation_id: self.correlation_id.clone(),
            outcome: RemoteWorkerOutcome::CollectiveRequest {
                attempt_id: self.attempt_id,
                request,
            },
        };
        if let Err(error) = super::worker_protocol::reply(
            self.connection.as_ref(),
            self.sender.as_ref(),
            self.limits,
            reply,
        )
        .await
        {
            self.pending.remove(key);
            return Err(transport_error(error));
        }
        match receiver.await.map_err(|_| {
            transport_error("remote collective completion channel closed before a response")
        })? {
            crate::protocol::CollectiveProcessResult::Completed { response } => Ok(response),
            crate::protocol::CollectiveProcessResult::Failed { message } => {
                Err(transport_error(message))
            }
        }
    }

    pub(super) fn complete(
        &self,
        context: &SpmdTaskContext,
        id: CollectiveId,
        sequence: CollectiveSequence,
        result: crate::protocol::CollectiveProcessResult,
    ) -> Result<(), String> {
        if context != &self.context {
            return Err(
                "remote collective completion differs from its admitted rank context".into(),
            );
        }
        self.pending
            .complete(CompletionKey { id, sequence }, result)
    }
}

pub(super) struct RemoteCollectiveService {
    context: SpmdTaskContext,
    channel: Arc<RemoteCollectiveChannel>,
    sequences: RefCell<BTreeMap<CollectiveId, u64>>,
}

impl RemoteCollectiveService {
    pub(super) fn new(channel: Arc<RemoteCollectiveChannel>) -> Self {
        Self {
            context: channel.context.clone(),
            channel,
            sequences: RefCell::new(BTreeMap::new()),
        }
    }
}

impl RuntimeCollectiveService for RemoteCollectiveService {
    fn context(&self) -> &SpmdTaskContext {
        &self.context
    }

    fn next_sequence(&self, id: CollectiveId) -> Result<CollectiveSequence, RuntimeError> {
        let mut sequences = self.sequences.borrow_mut();
        let sequence = sequences.entry(id).or_default();
        *sequence = sequence
            .checked_add(1)
            .ok_or_else(|| transport_error("remote collective sequence overflowed"))?;
        Ok(CollectiveSequence(*sequence))
    }

    fn execute(
        &self,
        request: CollectiveRequest,
    ) -> RuntimeServiceFuture<Result<CollectiveResponse, RuntimeError>> {
        let channel = Arc::clone(&self.channel);
        Box::pin(async move { channel.submit(request).await })
    }
}

fn transport_error(message: impl std::fmt::Display) -> RuntimeError {
    runmat_runtime::runtime_error::semantic_error(
        "RunMat:parallel:CollectiveTransport",
        message.to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(ordinal: u32, sequence: u64) -> CompletionKey {
        CompletionKey {
            id: CollectiveId {
                region: runmat_types::ParallelRegionId(runmat_types::RegionId {
                    function: runmat_types::ProgramFunctionId(1),
                    ordinal: 2,
                }),
                ordinal,
            },
            sequence: CollectiveSequence(sequence),
        }
    }

    #[tokio::test]
    async fn completion_registry_requires_the_exact_collective_identity() {
        let registry = CollectiveCompletionRegistry::default();
        let expected = key(7, 11);
        let (sender, receiver) = oneshot::channel();
        registry.register(expected, sender).unwrap();

        let mismatch = registry.complete(
            key(7, 12),
            crate::protocol::CollectiveProcessResult::Completed {
                response: CollectiveResponse::Complete,
            },
        );
        assert_eq!(
            mismatch.unwrap_err(),
            "remote collective completion identity is unknown or stale"
        );

        registry
            .complete(
                expected,
                crate::protocol::CollectiveProcessResult::Completed {
                    response: CollectiveResponse::Complete,
                },
            )
            .unwrap();
        assert!(matches!(
            receiver.await.unwrap(),
            crate::protocol::CollectiveProcessResult::Completed {
                response: CollectiveResponse::Complete
            }
        ));
    }
}
