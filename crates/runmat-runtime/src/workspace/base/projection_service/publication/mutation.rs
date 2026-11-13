//! Typed identity and origin for pre-effect publication preparation.

use super::WorkspacePublicationOrigin;
use crate::workspace::WorkspaceRevision;

/// The exact projection transition observed by a publication participant.
/// Only the owning Runtime service can issue this identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspacePublicationStamp {
    issuer: usize,
    revision: WorkspaceRevision,
    before_generation: u64,
    after_generation: u64,
}

impl WorkspacePublicationStamp {
    pub(in crate::workspace::base::projection_service) fn issued(
        issuer: usize,
        revision: WorkspaceRevision,
        before_generation: u64,
        after_generation: u64,
    ) -> Self {
        Self {
            issuer,
            revision,
            before_generation,
            after_generation,
        }
    }

    pub const fn revision(self) -> WorkspaceRevision {
        self.revision
    }

    pub const fn before_generation(self) -> u64 {
        self.before_generation
    }

    pub const fn after_generation(self) -> u64 {
        self.after_generation
    }

    /// A new transition must continue the exact previously staged projection,
    /// not another service at the same revision or an earlier pending prefix.
    pub fn continues(self, previous: Self) -> bool {
        self.issuer == previous.issuer
            && self.revision == previous.revision
            && self.before_generation == previous.after_generation
    }

    pub fn same_destination(self, other: Self) -> bool {
        self.issuer == other.issuer
            && self.revision == other.revision
            && self.after_generation == other.after_generation
    }
}

/// One prospective source effect, resolved against its authentic prepared
/// projection. Participants must finish borrowing the origin during prepare.
#[derive(Clone, Copy, Debug)]
pub struct WorkspacePublicationMutation<'a> {
    pub stamp: WorkspacePublicationStamp,
    pub origin: WorkspacePublicationOrigin<'a>,
}

impl<'a> WorkspacePublicationMutation<'a> {
    pub(in crate::workspace::base::projection_service) fn issued(
        stamp: WorkspacePublicationStamp,
        origin: WorkspacePublicationOrigin<'a>,
    ) -> Self {
        Self { stamp, origin }
    }
}
