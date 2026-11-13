//! Preserve operational causes at the statistics-options host materialization boundary.

use super::*;
use runmat_async::GpuGatherRetry;

pub(super) fn statset_materialization_error(source: RuntimeError) -> RuntimeError {
    let error = statset_error(&STATSET_ERROR_INVALID_ARGUMENT, source.message());
    retain_cause(error, source)
}

pub(super) fn statget_materialization_error(source: RuntimeError) -> RuntimeError {
    let error = statget_error(&STATGET_ERROR_INVALID_ARGUMENT, source.message());
    retain_cause(error, source)
}

fn retain_cause(mut error: RuntimeError, source: RuntimeError) -> RuntimeError {
    error.source = Some(Box::new(source));
    error.gpu_gather_retry = GpuGatherRetry::Never;
    error
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operational_failure_preserves_cause_and_cannot_request_replay() {
        for (remap, descriptor, builtin) in [
            (
                statset_materialization_error as fn(RuntimeError) -> RuntimeError,
                &STATSET_ERROR_INVALID_ARGUMENT,
                STATSET,
            ),
            (
                statget_materialization_error as fn(RuntimeError) -> RuntimeError,
                &STATGET_ERROR_INVALID_ARGUMENT,
                STATGET,
            ),
        ] {
            let source = build_runtime_error("provider read failed")
                .with_identifier("RunMat:ProviderRead")
                .with_phase("readback")
                .with_source(std::io::Error::other("device disconnected"))
                .build();
            let error = remap(source);
            assert_eq!(error.identifier(), descriptor.identifier);
            assert_eq!(error.context.builtin.as_deref(), Some(builtin));
            assert_eq!(error.gpu_gather_retry(), GpuGatherRetry::Never);
            let source = error.source.unwrap().downcast::<RuntimeError>().unwrap();
            assert_eq!(source.identifier(), Some("RunMat:ProviderRead"));
            assert_eq!(source.context.phase.as_deref(), Some("readback"));
            let device = source.source.unwrap().downcast::<std::io::Error>().unwrap();
            assert_eq!(device.kind(), std::io::ErrorKind::Other);
            assert_eq!(device.to_string(), "device disconnected");
        }
    }
}
