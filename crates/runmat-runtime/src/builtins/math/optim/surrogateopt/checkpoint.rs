//! Private selected checkpoint records; no executable authority comes from a
//! filename, serialized callback spelling or a previous invocation context.

pub(super) mod bits;
mod callbacks;
mod geometry;
pub(super) mod json;
mod lifecycle;
mod options;
pub(super) mod parallel;
mod record;
mod session;
pub(super) use lifecycle::{resume, start};
mod trials;
pub(super) use callbacks::ArchivedCallbacks;
pub(super) use session::Session;

pub(super) fn error(message: impl std::fmt::Display) -> crate::RuntimeError {
    let descriptor = &runmat_builtins::SURROGATEOPT_ERROR_CHECKPOINT;
    let mut builder = crate::build_runtime_error(format!("surrogateopt: {message}"))
        .with_builtin(super::NAME)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn with_source(
    message: impl std::fmt::Display,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> crate::RuntimeError {
    let descriptor = &runmat_builtins::SURROGATEOPT_ERROR_CHECKPOINT;
    let mut builder = crate::build_runtime_error(format!("surrogateopt: {message}"))
        .with_builtin(super::NAME)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .with_source(source);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
fn codec_error(source: crate::RuntimeError) -> crate::RuntimeError {
    if crate::runtime_error::error_has_control_cancellation_origin(&source) {
        source
    } else {
        with_source("invalid checkpoint record encoding", source)
    }
}
