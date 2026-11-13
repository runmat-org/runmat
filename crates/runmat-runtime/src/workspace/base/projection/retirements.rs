//! Fixed-size physical owners retired by one projection installation.
//!
//! The caller drops this carrier only after releasing both the projection
//! service and canonical Base borrows. No new collection or allocation is
//! needed at the visible publication boundary.

use std::collections::HashMap;

use super::super::delta::RetiredWorkspaceDeltaMutation;
use super::super::storage::ManagedWorkspaceStorage;
use super::super::values::{RootedWorkspaceEntries, RootedWorkspaceEntry};
use super::super::{WorkspaceChange, WorkspaceDelta};
use crate::workspace::GlobalBindingTarget;

#[derive(Default)]
pub(in crate::workspace::base) struct ProjectionRetirements {
    pub(in crate::workspace::base) entries: Option<RootedWorkspaceEntries>,
    pub(in crate::workspace::base) legacy_entry: Option<RootedWorkspaceEntry>,
    pub(in crate::workspace::base) globals: Option<HashMap<String, GlobalBindingTarget>>,
    pub(in crate::workspace::base) storage: Option<ManagedWorkspaceStorage>,
    pub(in crate::workspace::base) _projection_prefix: Option<ManagedWorkspaceStorage>,
    pub(in crate::workspace::base) canonical_prefix: Option<ManagedWorkspaceStorage>,
    pub(in crate::workspace::base) pending: Option<WorkspaceDelta>,
    pub(in crate::workspace::base) delta_mutation: Option<RetiredWorkspaceDeltaMutation>,
    pub(in crate::workspace::base) replaced_changes: [Option<WorkspaceChange>; 2],
}
