//! Canonical, session-lifetime state for a base workspace.
//!
//! Executors receive disposable projections and publish ordered deltas. They
//! never replace this owner from a completion snapshot.

mod commit;
mod delta;
mod error;
mod journal;
mod owner;
mod projection;
mod revision;
mod values;

pub use delta::{WorkspaceChange, WorkspaceChangeKind, WorkspaceDelta};
pub use error::BaseWorkspaceError;
pub use owner::{
    BaseWorkspaceCommit, BaseWorkspaceOwner, BaseWorkspaceReconciliation, BaseWorkspaceSnapshot,
};
pub use projection::BaseWorkspaceProjection;
pub use revision::WorkspaceRevision;

#[cfg(test)]
mod tests;
