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
mod service;
mod values;

pub use delta::{WorkspaceChange, WorkspaceChangeKind, WorkspaceDelta};
pub use error::BaseWorkspaceError;
pub use owner::{
    BaseWorkspaceCommit, BaseWorkspaceOwner, BaseWorkspaceReconciliation, BaseWorkspaceSnapshot,
};
pub use projection::BaseWorkspaceProjection;
pub use revision::WorkspaceRevision;
pub use service::base_workspace_error_to_runtime;
pub use service::BaseWorkspaceService;

#[cfg(test)]
mod authority_tests;
#[cfg(test)]
mod tests;
