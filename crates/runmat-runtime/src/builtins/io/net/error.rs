//! Preserve transport and provider causes behind the public network diagnostics.

use crate::runtime_error::RuntimeErrorSourceBound;
use crate::{build_runtime_error, GpuGatherRetry, RuntimeError};
use runmat_builtins::BuiltinErrorDescriptor;

pub(super) fn with_source(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: impl RuntimeErrorSourceBound,
) -> RuntimeError {
    let detail = detail.as_ref();
    let prefix = format!("{builtin}: ");
    let detail = detail.strip_prefix(&prefix).unwrap_or(detail);
    let mut builder = build_runtime_error(format!("{}: {detail}", error.message))
        .with_builtin(builtin)
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn with_source_message(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: impl RuntimeErrorSourceBound,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(builtin)
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{error::Error, io};

    #[test]
    fn public_identifier_and_message_keep_the_typed_socket_cause() {
        let source = io::Error::new(io::ErrorKind::ConnectionRefused, "peer refused");
        let error = with_source(
            "tcpclient",
            &runmat_builtins::TCPCLIENT_ERROR_CONNECT_FAILED,
            format!("tcpclient: unable to connect ({source})"),
            source,
        );
        assert_eq!(
            error.identifier(),
            runmat_builtins::TCPCLIENT_ERROR_CONNECT_FAILED.identifier
        );
        assert!(error.message().contains("unable to connect"));
        assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
        assert_eq!(
            Error::source(&error)
                .and_then(|source| source.downcast_ref::<io::Error>())
                .map(io::Error::kind),
            Some(io::ErrorKind::ConnectionRefused)
        );
    }
}
