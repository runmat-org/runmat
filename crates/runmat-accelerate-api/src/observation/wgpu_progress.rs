//! Typed device-progress port. This is not an arbitrary task executor: the only
//! submitted work is waiting for an exact WGPU submission away from the runtime.

/// Completion and its already-admitted callback backing claim. The claim is
/// deliberately outside the callback Box: it outlives consuming invocation,
/// deallocation, and unwind. No new allocation is made by this outer owner.
#[cfg(not(target_arch = "wasm32"))]
type CompletionCallback = Box<dyn FnOnce(Result<(), WgpuObservationProgressError>) + Send>;
#[cfg(target_arch = "wasm32")]
type CompletionCallback = Box<dyn FnOnce(Result<(), WgpuObservationProgressError>)>;

pub struct WgpuObservationCompletion {
    callback: Option<CompletionCallback>,
    _claims:
        [Option<std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>>; 2],
}

impl WgpuObservationCompletion {
    pub fn new(
        callback: CompletionCallback,
        claims: [Option<std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>>;
            2],
    ) -> Self {
        Self {
            callback: Some(callback),
            _claims: claims,
        }
    }

    pub fn finish(mut self, status: Result<(), WgpuObservationProgressError>) {
        if let Some(callback) = self.callback.take() {
            callback(status);
        }
    }
}

impl Drop for WgpuObservationCompletion {
    fn drop(&mut self) {
        if let Some(callback) = self.callback.take() {
            callback(Err(WgpuObservationProgressError::Stopped));
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WgpuObservationProgressError {
    Capacity,
    Stopped,
}
impl std::fmt::Display for WgpuObservationProgressError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Capacity => "WGPU progress capacity is exhausted",
            Self::Stopped => "WGPU progress owner has stopped",
        })
    }
}
impl std::error::Error for WgpuObservationProgressError {}

pub trait WgpuObservationProgress: Send + Sync {
    /// Reserve before allocating a fence or submitting/map-registering work.
    fn reserve(
        &self,
    ) -> Result<Box<dyn WgpuObservationProgressReservation>, WgpuObservationProgressError>;
}

pub trait WgpuObservationProgressReservation: Send {
    /// Nonblocking commit, called outside any device emission permit. Invoke
    /// `completion` exactly once after progress returns, or with an error if
    /// delivery/progress fails. The callback may acquire a fresh emission permit
    /// to unmap its private fence; map callbacks themselves must only record
    /// completion. A progress failure is not evidence of device loss.
    /// Dropping an uncommitted reservation releases its mailbox capacity.
    fn submit(
        self: Box<Self>,
        submission: wgpu::SubmissionIndex,
        completion: WgpuObservationCompletion,
    ) -> Result<(), WgpuObservationProgressError>;
}
