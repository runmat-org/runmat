use runmat_builtins::BuiltinErrorDescriptor;

use crate::context::GraphicsFlushDisposition;
use crate::execution::ExecutionServiceError;
use crate::{build_runtime_error, RuntimeError};

#[derive(Debug)]
pub(super) struct GraphicsFlushEffectSource {
    disposition: GraphicsFlushDisposition,
    source: Option<Box<runmat_async::runtime_error::RuntimeErrorSource>>,
}

impl GraphicsFlushEffectSource {
    pub(super) fn disposition(&self) -> GraphicsFlushDisposition {
        self.disposition
    }
}

impl std::fmt::Display for GraphicsFlushEffectSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "graphics flush outcome is {:?}",
            self.disposition
        )
    }
}

impl std::error::Error for GraphicsFlushEffectSource {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|source| source.as_ref() as &(dyn std::error::Error + 'static))
    }
}

pub(super) fn cancelled_flush(
    source: ExecutionServiceError,
    disposition: GraphicsFlushDisposition,
) -> RuntimeError {
    let error = build_runtime_error(format!("drawnow: {source}"))
        .with_identifier("RunMat:Cancelled")
        .with_builtin("drawnow")
        .with_source(source)
        .build();
    attach_disposition(error, disposition)
}

/// Retain the original diagnostic and downcastable cause while adding the
/// effect disposition observed at the exact cancellation/error boundary.
pub(super) fn attach_disposition(
    mut error: RuntimeError,
    disposition: GraphicsFlushDisposition,
) -> RuntimeError {
    let source = error.source.take();
    error.source = Some(Box::new(GraphicsFlushEffectSource {
        disposition,
        source,
    }));
    error
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
