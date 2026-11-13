//! Shared immutable name and value owners for workspace and session storage.
//!
//! Physical claims follow the actual owner allocations. A shared Value owner
//! does not, by itself, register GC roots or authorize cloning arbitrary
//! language payloads into finite workspace storage.

mod name;
mod value;

pub use name::WorkspaceName;
pub use value::WorkspaceValueReadLease;
pub(crate) use value::{
    is_allocation_free_workspace_alias, WorkspaceRetainedValue, WorkspaceValueLease,
};
