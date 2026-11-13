//! Canonical, session-lifetime state for a base workspace.
//!
//! Executors receive disposable projections and publish ordered deltas. They
//! never replace this owner from a completion snapshot.

mod binding_names;
mod commit;
mod delta;
mod error;
mod journal;
mod logical;
mod owner;
mod prepared_write;
mod projection;
mod projection_service;
mod revision;
mod service;
mod storage;
mod values;

pub use crate::workspace::retained::WorkspaceName;
pub(crate) use crate::workspace::retained::WorkspaceRetainedValue;
pub use binding_names::BaseWorkspaceBindingNames;
pub use delta::{WorkspaceChange, WorkspaceChangeKind, WorkspaceDelta};
pub use error::BaseWorkspaceError;
pub use logical::{
    assign_logical_base_value, assign_logical_base_value_ref_with_context,
    assign_logical_base_value_with_context, lookup_logical_base_value,
};
pub use owner::{
    BaseWorkspaceCommit, BaseWorkspaceOwner, BaseWorkspaceReconciliation, BaseWorkspaceSnapshot,
};
pub(crate) use prepared_write::PreparedWorkspaceWrite;
pub use projection::BaseWorkspaceProjection;
pub use projection_service::{
    BaseWorkspaceProjectionService, PreparedWorkspaceCompletion, WorkspaceProjectionPublication,
    WorkspacePublicationBatch, WorkspacePublicationMutation, WorkspacePublicationObserver,
    WorkspacePublicationOrigin, WorkspacePublicationProtocol, WorkspacePublicationStamp,
    WorkspacePublicationToken,
};
pub use revision::WorkspaceRevision;
pub use service::base_workspace_error_to_runtime;
pub use service::BaseWorkspaceService;

#[cfg(test)]
mod authority_tests;
#[cfg(test)]
mod tests;
