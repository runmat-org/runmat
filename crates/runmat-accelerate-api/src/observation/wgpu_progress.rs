//! Typed device-progress port. This is not an arbitrary task executor: the only
//! submitted work is waiting for an exact WGPU submission away from the runtime.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WgpuObservationProgressError {
    Capacity,
    Stopped,
}

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
        completion: Box<dyn FnOnce(Result<(), WgpuObservationProgressError>) + Send>,
    ) -> Result<(), WgpuObservationProgressError>;
}
