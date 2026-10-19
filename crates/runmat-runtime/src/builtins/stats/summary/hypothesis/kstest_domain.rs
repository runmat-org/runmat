//! One-sample KS result and reference-distribution domain values.
use super::*;
pub(super) fn kstest_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(KSTEST_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn kstest_invalid_argument(message: impl Into<String>) -> RuntimeError {
    kstest_error(message, &KSTEST_ERROR_INVALID_ARGUMENT)
}

#[derive(Clone, Debug)]
pub(super) enum CdfSpec {
    StandardNormal,
    Table(Vec<(f64, f64)>),
}

#[derive(Clone, Debug)]
pub(super) struct KstestOptions {
    pub(super) alpha: f64,
    pub(super) tail: KstestTail,
    pub(super) cdf: CdfSpec,
}

impl Default for KstestOptions {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            tail: KstestTail::Unequal,
            cdf: CdfSpec::StandardNormal,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct KstestEvaluation {
    pub(super) h: Value,
    pub(super) p: Value,
    pub(super) ksstat: Value,
    pub(super) cv: Value,
}
