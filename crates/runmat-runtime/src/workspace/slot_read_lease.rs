//! Immutable source-value ownership shared by workspace execution backends.
//!
//! A lease retains one existing value generation and its GC roots. It does
//! not copy the language payload or mutate the originating binding.

use crate::call::lexical::LexicalBindingReadLease;
use crate::context::RuntimeContext;
use crate::execution::RetainedValueReadLease;
use crate::RuntimeError;
use runmat_value::{SharedValueGeneration, Value};

use super::WorkspaceValueReadLease;

#[derive(Clone, Debug)]
pub enum WorkspaceSlotReadLease {
    Workspace(WorkspaceValueReadLease),
    Lexical(LexicalBindingReadLease),
    Owned(RetainedValueReadLease),
}

impl WorkspaceSlotReadLease {
    /// Move an already-owned Value into one immutable generation under the
    /// current operation. This admits only the new shared header and roots;
    /// it does not claim that arbitrary preexisting payload backing was
    /// allocated by this operation.
    pub fn prepare_owned(value: Value, context: &RuntimeContext) -> Result<Self, RuntimeError> {
        RetainedValueReadLease::prepare_owned(value, context).map(Self::Owned)
    }

    pub fn value(&self) -> &Value {
        match self {
            Self::Workspace(lease) => lease.value(),
            Self::Lexical(lease) => lease.value(),
            Self::Owned(lease) => lease.value(),
        }
    }

    pub(crate) fn shared_generation(&self) -> SharedValueGeneration {
        match self {
            Self::Workspace(lease) => lease.shared_generation(),
            Self::Lexical(lease) => lease.shared_generation(),
            Self::Owned(lease) => lease.shared_generation(),
        }
    }

    pub(crate) fn shared_roots(&self) -> Option<crate::execution::RootedValueHandles> {
        match self {
            Self::Workspace(lease) => lease.shared_roots(),
            Self::Lexical(lease) => Some(lease.shared_roots()),
            Self::Owned(lease) => Some(lease.shared_roots()),
        }
    }

    /// Compare immutable source generations without comparing language values
    /// or cloning their payloads. A retained old lease prevents address reuse
    /// until the comparison has completed.
    pub fn shares_generation_with(&self, other: &Self) -> bool {
        self.shared_generation().ptr_eq(&other.shared_generation())
    }
}

impl From<WorkspaceValueReadLease> for WorkspaceSlotReadLease {
    fn from(value: WorkspaceValueReadLease) -> Self {
        Self::Workspace(value)
    }
}

impl From<LexicalBindingReadLease> for WorkspaceSlotReadLease {
    fn from(value: LexicalBindingReadLease) -> Self {
        Self::Lexical(value)
    }
}
