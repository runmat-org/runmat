use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use runmat_execution::{
    CollectiveRequest, CollectiveResponse, CollectiveSequence, SpmdTaskContext,
};
use runmat_process_host::ipc::private_control::{ControlFrameWriter, ControlWriteError};
use runmat_runtime::context::{RuntimeCollectiveService, RuntimeServiceFuture};
use runmat_runtime::RuntimeError;
use runmat_types::CollectiveId;

use crate::protocol::{CollectiveProcessResult, WorkerDriverMessage, WorkerProcessMessage};

type ParentFrames = tokio::sync::mpsc::Receiver<runmat_process_host::ProcessHostResult<Vec<u8>>>;

#[derive(Clone)]
pub(super) struct WorkerControlChannel {
    reader: Rc<tokio::sync::Mutex<ParentFrames>>,
    writer: ControlFrameWriter,
}

impl WorkerControlChannel {
    pub(super) fn new(reader: ParentFrames, writer: ControlFrameWriter) -> Self {
        Self {
            reader: Rc::new(tokio::sync::Mutex::new(reader)),
            writer,
        }
    }

    pub(super) fn writer(&self) -> &ControlFrameWriter {
        &self.writer
    }

    pub(super) async fn read_payload(
        &self,
    ) -> Result<Vec<u8>, runmat_process_host::ProcessHostError> {
        let mut reader = self.reader.lock().await;
        next_frame(&mut reader).await
    }

    pub(super) async fn write_payload(
        &self,
        payload: Vec<u8>,
    ) -> Result<(), runmat_process_host::ProcessHostError> {
        self.writer
            .write_owned(payload)
            .await
            .map_err(control_error)
    }

    async fn exchange(
        &self,
        payload: Vec<u8>,
    ) -> Result<Vec<u8>, runmat_process_host::ProcessHostError> {
        let mut reader = self.reader.lock().await;
        self.writer
            .write_owned(payload)
            .await
            .map_err(control_error)?;
        next_frame(&mut reader).await
    }
}

async fn next_frame(
    frames: &mut ParentFrames,
) -> Result<Vec<u8>, runmat_process_host::ProcessHostError> {
    frames.recv().await.unwrap_or_else(|| {
        Err(runmat_process_host::ProcessHostError::Protocol(
            "parent request pipe closed".into(),
        ))
    })
}

pub(super) struct WorkerCollectiveService {
    context: SpmdTaskContext,
    channel: WorkerControlChannel,
    sequences: RefCell<BTreeMap<CollectiveId, u64>>,
}

impl WorkerCollectiveService {
    pub(super) fn new(context: SpmdTaskContext, channel: WorkerControlChannel) -> Self {
        Self {
            context,
            channel,
            sequences: RefCell::new(BTreeMap::new()),
        }
    }
}

impl RuntimeCollectiveService for WorkerCollectiveService {
    fn context(&self) -> &SpmdTaskContext {
        &self.context
    }

    fn next_sequence(&self, id: CollectiveId) -> Result<CollectiveSequence, RuntimeError> {
        if id.region != self.context.region {
            return Err(error(
                "collective identity does not belong to this SPMD worker context",
            ));
        }
        let mut sequences = self.sequences.borrow_mut();
        let sequence = sequences.entry(id).or_default();
        *sequence = sequence
            .checked_add(1)
            .ok_or_else(|| error("collective invocation sequence overflowed"))?;
        Ok(CollectiveSequence(*sequence))
    }

    fn execute(
        &self,
        request: CollectiveRequest,
    ) -> RuntimeServiceFuture<Result<CollectiveResponse, RuntimeError>> {
        if request.context != self.context {
            return Box::pin(async {
                Err(error(
                    "collective request does not belong to this SPMD worker context",
                ))
            });
        }
        let channel = self.channel.clone();
        Box::pin(async move {
            let context = request.context.clone();
            let id = request.id;
            let sequence = request.sequence;
            let payload = serde_json::to_vec(&WorkerProcessMessage::CollectiveRequest { request })
                .map_err(|failure| error(failure.to_string()))?;
            let payload = channel
                .exchange(payload)
                .await
                .map_err(|failure| error(failure.to_string()))?;
            let completion: WorkerDriverMessage =
                serde_json::from_slice(&payload).map_err(|failure| error(failure.to_string()))?;
            let WorkerDriverMessage::CollectiveCompletion {
                context: completed_context,
                id: completed_id,
                sequence: completed_sequence,
                result,
            } = completion
            else {
                return Err(error("collective received an unrelated driver control"));
            };
            if completed_context != context || completed_id != id || completed_sequence != sequence
            {
                return Err(error(
                    "collective driver response differs from the pending request identity",
                ));
            }
            match result {
                CollectiveProcessResult::Completed { response } => Ok(response),
                CollectiveProcessResult::Failed { message } => Err(error(message)),
            }
        })
    }
}

fn control_error(error: ControlWriteError) -> runmat_process_host::ProcessHostError {
    runmat_process_host::ProcessHostError::Io(std::io::Error::other(error))
}

fn error(message: impl Into<String>) -> RuntimeError {
    runmat_runtime::runtime_error::semantic_error(
        "RunMat:parallel:CollectiveTransport",
        message.into(),
    )
}
