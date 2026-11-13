//! One normalized dynamic-workspace completion transaction.
//!
//! The source is deliberately indexed and repeatable. Runtime owners first
//! inspect the complete final action set, then admit and prepare every owner
//! before installing any part of the completion. Callers keep the indexed
//! backing alive through the synchronous transaction; implementations must
//! not retain the borrowed actions.

use runmat_value::Value;

use crate::context::RuntimeContext;
use crate::RuntimeError;

use super::WorkspaceRevision;
use super::{GlobalBindingTarget, WorkspaceSlotReadLease};

#[derive(Clone, Debug)]
pub enum WorkspaceCompletionValue<'a> {
    Borrowed(&'a Value),
    Retained(WorkspaceSlotReadLease),
}

impl WorkspaceCompletionValue<'_> {
    pub fn value(&self) -> &Value {
        match self {
            Self::Borrowed(value) => value,
            Self::Retained(value) => value.value(),
        }
    }
}

impl<'a> From<&'a Value> for WorkspaceCompletionValue<'a> {
    fn from(value: &'a Value) -> Self {
        Self::Borrowed(value)
    }
}

#[derive(Clone, Debug)]
pub enum WorkspaceCompletionAction<'a> {
    AssignLocal {
        name: &'a str,
        value: WorkspaceCompletionValue<'a>,
    },
    BindGlobal {
        name: &'a str,
        target: &'a GlobalBindingTarget,
    },
    UnbindGlobal {
        name: &'a str,
    },
    Remove {
        name: &'a str,
    },
}

impl<'a> WorkspaceCompletionAction<'a> {
    pub fn name(&self) -> &'a str {
        match self {
            Self::AssignLocal { name, .. }
            | Self::BindGlobal { name, .. }
            | Self::UnbindGlobal { name }
            | Self::Remove { name } => name,
        }
    }
}

/// Stable, normalized action storage owned by the caller.
///
/// `action(index)` must return the same action for every pass during one
/// completion call. Names must be unique and in deterministic order. The
/// receiving owner validates both properties before it allocates or mutates.
pub trait WorkspaceCompletionSource {
    fn len(&self) -> usize;

    fn action(&self, index: usize) -> Option<WorkspaceCompletionAction<'_>>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A prepared source-frame participant in one whole-workspace publication.
///
/// The caller owns its frame and all admitted backing. Checkpoint must finish
/// every fallible authority check before Runtime locks Base/lexical stores.
/// `install_prevalidated` then performs only an infallible, allocation-free
/// state swap: no callbacks, language execution, or owner retirement. It
/// returns any displaced owner so the coordinator can release it only after
/// Base/GC guards and publication observers have finished.
pub trait WorkspaceCompletionParticipant {
    type Retirement;

    fn checkpoint_before_install(&mut self, context: &RuntimeContext) -> Result<(), RuntimeError>;

    fn install_prevalidated(&mut self) -> Self::Retirement;
}

/// Revision captured with the exact workspace target before its child runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkspaceCompletionIdentity {
    revision: WorkspaceRevision,
    pending_generation: u64,
    issuer: usize,
}

impl WorkspaceCompletionIdentity {
    pub(crate) const fn issued(
        revision: WorkspaceRevision,
        pending_generation: u64,
        issuer: usize,
    ) -> Self {
        Self {
            revision,
            pending_generation,
            issuer,
        }
    }

    pub const fn revision(self) -> WorkspaceRevision {
        self.revision
    }

    /// Issue a composite completion identity at a retained service address,
    /// preserving the lower captured revision and pending generation. The
    /// composite service must separately validate that lower capture before
    /// it applies any completion action.
    pub fn reissued_by(self, service: &dyn crate::context::RuntimeWorkspaceService) -> Self {
        Self::issued(
            self.revision,
            self.pending_generation,
            service as *const dyn crate::context::RuntimeWorkspaceService as *const () as usize,
        )
    }

    pub(crate) const fn pending_generation(self) -> u64 {
        self.pending_generation
    }

    pub(crate) const fn issuer(self) -> usize {
        self.issuer
    }
}
