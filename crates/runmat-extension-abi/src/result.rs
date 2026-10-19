use crate::{RunMatErrorView, RunMatValueHandle};

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunMatStatusCode {
    Ok = 0,
    InvalidArgument = 1,
    Unsupported = 2,
    Cancelled = 3,
    Failed = 4,
    Panic = 5,
    AbiMismatch = 6,
    StaleHandle = 7,
    AffinityViolation = 8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RunMatExtensionResult {
    pub status: RunMatStatusCode,
    pub outputs: *const RunMatValueHandle,
    pub output_count: usize,
    pub error: RunMatErrorView,
}
