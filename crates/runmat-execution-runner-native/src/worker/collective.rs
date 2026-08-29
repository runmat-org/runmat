use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use runmat_execution::{
    CollectiveRequest, CollectiveResponse, CollectiveSequence, SpmdTaskContext,
};
use runmat_process_host::ipc::{read_payload, write_payload, FrameLimits};
use runmat_runtime::context::{RuntimeCollectiveService, RuntimeServiceFuture};
use runmat_runtime::RuntimeError;
use runmat_types::CollectiveId;
use tokio::io::{BufReader, BufWriter, Stdin, Stdout};

use crate::protocol::{CollectiveProcessResult, WorkerDriverMessage, WorkerProcessMessage};

#[derive(Clone)]
pub(super) struct StdioCollectiveChannel {
    io: Rc<tokio::sync::Mutex<StdioCollectiveIo>>,
    limits: FrameLimits,
}

struct StdioCollectiveIo {
    reader: BufReader<Stdin>,
    writer: BufWriter<Stdout>,
}

impl StdioCollectiveChannel {
    pub(super) fn new(
        reader: BufReader<Stdin>,
        writer: BufWriter<Stdout>,
        limits: FrameLimits,
    ) -> Self {
        Self {
            io: Rc::new(tokio::sync::Mutex::new(StdioCollectiveIo {
                reader,
                writer,
            })),
            limits,
        }
    }

    pub(super) async fn read_payload(
        &self,
    ) -> Result<Vec<u8>, runmat_process_host::ProcessHostError> {
        let mut io = self.io.lock().await;
        read_payload(&mut io.reader, self.limits).await
    }

    pub(super) async fn write_payload(
        &self,
        payload: &[u8],
    ) -> Result<(), runmat_process_host::ProcessHostError> {
        let mut io = self.io.lock().await;
        write_payload(&mut io.writer, payload, self.limits).await
    }

    async fn exchange(
        &self,
        payload: &[u8],
    ) -> Result<Vec<u8>, runmat_process_host::ProcessHostError> {
        let mut io = self.io.lock().await;
        write_payload(&mut io.writer, payload, self.limits).await?;
        read_payload(&mut io.reader, self.limits).await
    }
}

pub(super) struct StdioCollectiveService {
    context: SpmdTaskContext,
    channel: StdioCollectiveChannel,
    sequences: RefCell<BTreeMap<CollectiveId, u64>>,
}

impl StdioCollectiveService {
    pub(super) fn new(context: SpmdTaskContext, channel: StdioCollectiveChannel) -> Self {
        Self {
            context,
            channel,
            sequences: RefCell::new(BTreeMap::new()),
        }
    }
}

impl RuntimeCollectiveService for StdioCollectiveService {
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
                .exchange(&payload)
                .await
                .map_err(|failure| error(failure.to_string()))?;
            let completion: WorkerDriverMessage =
                serde_json::from_slice(&payload).map_err(|failure| error(failure.to_string()))?;
            let WorkerDriverMessage::CollectiveCompletion {
                context: completed_context,
                id: completed_id,
                sequence: completed_sequence,
                result,
            } = completion;
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

fn error(message: impl Into<String>) -> RuntimeError {
    runmat_runtime::runtime_error::semantic_error(
        "RunMat:parallel:CollectiveTransport",
        message.into(),
    )
}
