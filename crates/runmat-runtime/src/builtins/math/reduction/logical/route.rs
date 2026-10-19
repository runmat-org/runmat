//! Domain execution capability after the existing option parser has run.
use crate::builtins::common::spec::ReductionNaN;
use runmat_builtins::LogicalReductionKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LogicalReductionRoute {
    HostExact,
    ProviderEligible,
}

pub(super) fn select(kind: LogicalReductionKind, nan_mode: ReductionNaN) -> LogicalReductionRoute {
    match (kind, nan_mode) {
        // Preserve the existing exact omit-NaN path for any. Do not upload host
        // data only to gather it immediately; resident inputs still gather.
        (LogicalReductionKind::Any, ReductionNaN::Omit) => LogicalReductionRoute::HostExact,
        _ => LogicalReductionRoute::ProviderEligible,
    }
}
