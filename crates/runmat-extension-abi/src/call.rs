use crate::{RunMatCancellationToken, RunMatExtensionContext, RunMatValueHandle};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RunMatExtensionCall {
    pub context: *mut RunMatExtensionContext,
    pub arguments: *const RunMatValueHandle,
    pub argument_count: usize,
    pub requested_outputs: usize,
    pub cancellation: *const RunMatCancellationToken,
}
