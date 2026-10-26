//! Shared cumulative invocation boundaries, not a second numerical engine.
pub(in crate::builtins::math::reduction) mod admission;
pub(in crate::builtins::math::reduction) mod options;
pub(in crate::builtins::math::reduction) mod provider;
pub(in crate::builtins::math::reduction) mod output;
pub(in crate::builtins::math::reduction) mod complex;
use runmat_builtins::CumulativeKind;
pub(in crate::builtins::math::reduction) fn failure(
    kind: CumulativeKind,
    detail: impl std::fmt::Display,
) -> crate::RuntimeError {
    let entry = kind.catalog_entry();
    let mut error = crate::runtime_descriptor_error_with_detail(
        entry.identity.name,
        &entry.descriptor.errors[2],
        detail.to_string(),
    );
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}

pub(in crate::builtins::math::reduction) mod geometry;
