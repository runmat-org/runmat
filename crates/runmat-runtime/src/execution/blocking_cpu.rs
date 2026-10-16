//! Host-composed admission to one bounded native CPU-worker lane. Requests
//! retain their own inputs and managed reservations through physical exit.

#[cfg(not(target_arch = "wasm32"))]
use std::rc::Rc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, OnceLock};

#[cfg(not(target_arch = "wasm32"))]
use futures::channel::mpsc;
#[cfg(not(target_arch = "wasm32"))]
use futures::{SinkExt, StreamExt};
#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::Mutex;

use crate::context::RuntimeServiceFuture;

use super::ExecutionServiceError;

pub type OwnedCpuJob = Box<dyn FnOnce() + Send + 'static>;

pub trait RuntimeBlockingCpuService {
    /// Wait for bounded queue capacity, then transfer the entire owned job.
    /// Dropping the admission future before acceptance drops the job and its
    /// reservation. Once accepted, caller cancellation cannot retire it.
    fn submit(&self, job: OwnedCpuJob) -> RuntimeServiceFuture<Result<(), ExecutionServiceError>>;
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeBlockingCpuService;

#[cfg(not(target_arch = "wasm32"))]
static NATIVE_OWNER: OnceLock<Result<Arc<Mutex<mpsc::Sender<OwnedCpuJob>>>, String>> =
    OnceLock::new();

#[cfg(not(target_arch = "wasm32"))]
fn native_owner() -> Result<Arc<Mutex<mpsc::Sender<OwnedCpuJob>>>, ExecutionServiceError> {
    let owner = NATIVE_OWNER.get_or_init(|| {
        // One executing and one queued job; other callers await admission
        // without blocking an executor thread or spuriously failing busy.
        // futures::mpsc reserves one sender slot in addition to the explicit
        // buffer. With the single mutex-held sender, zero buffer means at
        // most one queued job behind the one physically executing.
        let (send, mut receive) = mpsc::channel::<OwnedCpuJob>(0);
        std::thread::Builder::new()
            .name("runmat-blocking-cpu".into())
            .spawn(move || {
                futures::executor::block_on(async move {
                    while let Some(job) = receive.next().await {
                        job();
                    }
                });
            })
            .map_err(|error| error.to_string())?;
        Ok(Arc::new(Mutex::new(send)))
    });
    owner.as_ref().map(Arc::clone).map_err(|message| {
        ExecutionServiceError::Infrastructure(format!(
            "owned CPU worker could not start: {message}"
        ))
    })
}

#[cfg(not(target_arch = "wasm32"))]
impl RuntimeBlockingCpuService for NativeBlockingCpuService {
    fn submit(&self, job: OwnedCpuJob) -> RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
        match native_owner() {
            Ok(sender) => Box::pin(async move {
                let mut sender = sender.lock().await;
                sender
                    .send(job)
                    .await
                    .map_err(|_| ExecutionServiceError::WorkerLost("owned CPU worker ended".into()))
            }),
            Err(error) => Box::pin(async move { Err(error) }),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn native_default_blocking_cpu_service() -> Rc<dyn RuntimeBlockingCpuService> {
    Rc::new(NativeBlockingCpuService)
}
