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
    /// Nonblocking commit. Implementations invoke failure_cleanup if delivery or
    /// device progress fails; it fails observation and safely cancels mapping,
    /// never frees an allocation still retained by a callback. Dropping an
    /// uncommitted reservation releases its mailbox capacity.
    fn submit(
        self: Box<Self>,
        submission: wgpu::SubmissionIndex,
        failure_cleanup: Box<dyn FnOnce() + Send>,
    ) -> Result<(), WgpuObservationProgressError>;
}
