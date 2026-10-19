use runmat_builtins::BuiltinErrorDescriptor;

use crate::execution::ExecutionServiceError;
use crate::{build_runtime_error, RuntimeError};

pub(super) use crate::context::attach_graphics_disposition as attach_disposition;
#[cfg(test)]
pub(super) use crate::context::GraphicsFlushEffectSource;

pub(super) fn cancelled_flush(source: ExecutionServiceError) -> RuntimeError {
    build_runtime_error(format!("drawnow: {source}"))
        .with_identifier("RunMat:Cancelled")
        .with_builtin("drawnow")
        .with_source(source)
        .build()
}

pub(super) fn drawnow_error(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", descriptor.message, detail.as_ref()))
        .with_builtin("drawnow");
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
