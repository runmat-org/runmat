//! Shape-search adapters over the common accepted volume state.
use super::*;
use crate::cdt::insertion::topology_access::TetrahedralTopology;
use crate::cdt::topology::VolumeTopologyRead;
use crate::cdt::volume_quality::working::PreparedLocalQuality;
pub(super) use crate::cdt::volume_state::{SearchState, WorkingState, WorkingTrial};
use crate::cdt::volume_state::{VolumeStateError, VolumeStateErrorKind, VolumeStateOptions};
pub(in crate::cdt) mod rejected;
mod seed;
pub(super) mod treatment;
#[cfg(test)]
use crate::cdt::mutation::{TopologyEdit, TopologyMutation};
pub(super) use seed::{CaptureSeed, WorkingSeed};
#[cfg(test)]
use std::collections::BTreeSet;
#[cfg(test)]
mod tests;

impl From<DelaunayVolumeSliverOptions> for VolumeStateOptions {
    fn from(options: DelaunayVolumeSliverOptions) -> Self {
        Self {
            insertion: options.insertion,
            cancellation_check_interval: options.cancellation_check_interval,
        }
    }
}
impl From<VolumeStateError> for DelaunayVolumeSliverError {
    fn from(failure: VolumeStateError) -> Self {
        let kind = match failure.kind {
            VolumeStateErrorKind::Accounting(cause) => {
                DelaunayVolumeSliverErrorKind::Accounting(cause)
            }
            VolumeStateErrorKind::InvalidOptions => DelaunayVolumeSliverErrorKind::InvalidOptions,
            VolumeStateErrorKind::InvalidInput => DelaunayVolumeSliverErrorKind::InvalidInput,
            VolumeStateErrorKind::InvalidTopology => DelaunayVolumeSliverErrorKind::InvalidTopology,
            VolumeStateErrorKind::InvalidQuality => DelaunayVolumeSliverErrorKind::InvalidQuality,
            VolumeStateErrorKind::ResourceLimit => DelaunayVolumeSliverErrorKind::ResourceLimit,
            VolumeStateErrorKind::InvalidProvenance => {
                DelaunayVolumeSliverErrorKind::InvalidProvenance
            }
            VolumeStateErrorKind::Cancelled => DelaunayVolumeSliverErrorKind::Cancelled,
        };
        error(kind, failure.reason)
    }
}
