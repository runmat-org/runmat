use crate::backend::SerialBackend;

/// An explicit in-process backend uses the serial correctness adapter. Hosts
/// must advertise that it does not provide process isolation.
pub type InProcessBackend<F> = SerialBackend<F>;
