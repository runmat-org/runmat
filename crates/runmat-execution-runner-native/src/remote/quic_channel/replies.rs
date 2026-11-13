//! One channel owns request registration, terminal closure and original failure delivery.
use super::RemoteWorkerReply;
use crate::NativeExecutionError;
use runmat_execution::identity::AttemptId;
use runmat_execution_transport_native::frame::FrameKind;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{oneshot, watch};

pub(super) struct PendingReply {
    pub frame_kind: FrameKind,
    pub attempt_id: Option<AttemptId>,
    pub sender: oneshot::Sender<Result<RemoteWorkerReply, Arc<NativeExecutionError>>>,
    pub activity: watch::Sender<tokio::time::Instant>,
}

#[derive(Default)]
pub(super) struct ReplyRegistry {
    entries: HashMap<String, PendingReply>,
    closed: Option<Arc<NativeExecutionError>>,
}

impl ReplyRegistry {
    pub fn register(
        &mut self,
        correlation: String,
        reply: PendingReply,
    ) -> Result<(), Arc<NativeExecutionError>> {
        if let Some(error) = &self.closed {
            return Err(Arc::clone(error));
        }
        self.entries.insert(correlation, reply);
        Ok(())
    }

    pub fn get(&self, correlation: &str) -> Option<&PendingReply> {
        self.entries.get(correlation)
    }

    pub fn remove(&mut self, correlation: &str) -> Option<PendingReply> {
        self.entries.remove(correlation)
    }

    pub fn fail(&mut self, error: NativeExecutionError) -> Arc<NativeExecutionError> {
        let original = Arc::clone(self.closed.get_or_insert_with(|| Arc::new(error)));
        for (_, pending) in self.entries.drain() {
            let _ = pending.sender.send(Err(Arc::clone(&original)));
        }
        original
    }
}
