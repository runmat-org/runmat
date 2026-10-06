//! Canonical, session-lifetime state for a base workspace.
//!
//! Executors receive disposable projections and publish ordered deltas. They
//! never replace this owner from a completion snapshot.

mod delta;
mod error;
mod owner;
mod projection;
mod revision;
mod values;

pub use delta::{WorkspaceChange, WorkspaceDelta, MAX_WORKSPACE_DELTA_CHANGES};
pub use error::BaseWorkspaceError;
pub use owner::{
    BaseWorkspaceCommit, BaseWorkspaceOwner, BaseWorkspaceReconciliation, BaseWorkspaceSnapshot,
    MAX_BASE_WORKSPACE_ENTRIES,
};
pub use projection::BaseWorkspaceProjection;
pub use revision::WorkspaceRevision;

#[cfg(test)]
mod tests;
