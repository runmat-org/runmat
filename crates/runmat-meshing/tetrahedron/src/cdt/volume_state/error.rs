use crate::cdt::{
    DelaunayInsertionError, DelaunayInsertionErrorKind, DelaunayVolumeQualityError,
    DelaunayVolumeQualityErrorKind,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::cdt) enum VolumeStateErrorKind {
    Accounting(runmat_simulation_usage::LedgerError),
    InvalidOptions,
    InvalidInput,
    InvalidTopology,
    InvalidQuality,
    InvalidProvenance,
    ResourceLimit,
    Cancelled,
}
#[derive(Debug)]
pub(in crate::cdt) struct VolumeStateError {
    pub(in crate::cdt) kind: VolumeStateErrorKind,
    pub(in crate::cdt) reason: String,
}
impl std::fmt::Display for VolumeStateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "volume working state {:?}: {}", self.kind, self.reason)
    }
}
impl std::error::Error for VolumeStateError {}
pub(super) fn error(kind: VolumeStateErrorKind, reason: impl Into<String>) -> VolumeStateError {
    VolumeStateError {
        kind,
        reason: reason.into(),
    }
}
pub(super) fn quality_error(failure: DelaunayVolumeQualityError) -> VolumeStateError {
    let kind = match failure.kind {
        DelaunayVolumeQualityErrorKind::Accounting(cause) => {
            VolumeStateErrorKind::Accounting(cause)
        }
        DelaunayVolumeQualityErrorKind::ResourceLimit => VolumeStateErrorKind::ResourceLimit,
        DelaunayVolumeQualityErrorKind::Cancelled => VolumeStateErrorKind::Cancelled,
        DelaunayVolumeQualityErrorKind::InvalidOptions => VolumeStateErrorKind::InvalidOptions,
        DelaunayVolumeQualityErrorKind::InvalidTopology => VolumeStateErrorKind::InvalidTopology,
        DelaunayVolumeQualityErrorKind::InvalidMetric
        | DelaunayVolumeQualityErrorKind::InvalidMetricContext => {
            VolumeStateErrorKind::InvalidProvenance
        }
        DelaunayVolumeQualityErrorKind::InvalidQuality
        | DelaunayVolumeQualityErrorKind::NumericalFailure => VolumeStateErrorKind::InvalidQuality,
    };
    error(kind, failure.to_string())
}
pub(super) fn insertion_error(failure: DelaunayInsertionError) -> VolumeStateError {
    let kind = match failure.kind {
        DelaunayInsertionErrorKind::ResourceLimit => VolumeStateErrorKind::ResourceLimit,
        DelaunayInsertionErrorKind::Cancelled => VolumeStateErrorKind::Cancelled,
        _ => VolumeStateErrorKind::InvalidTopology,
    };
    error(kind, failure.to_string())
}
