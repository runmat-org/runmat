//! Persistent, exact-admitted workspace bindings and typed global targets.
//!
//! This is a physical binding owner, not a publication journal or a second
//! namespace authority. A caller prepares a replacement offstate, validates
//! its own revision while holding the canonical mutation gate, and moves the
//! displaced tree out of that gate before it can retire language values.

mod mutation;
mod node;

use crate::context::host_allocation::{HostAllocationError, HostOutputPermit};
use crate::context::RuntimeContext;
use crate::workspace::retained::WorkspaceName;
use crate::workspace::GlobalBindingTarget;

use super::super::values::RootedWorkspaceEntry;
use node::{Link, Node};

/// One live namespace state for a workspace spelling. A global binding can
/// exist even when its separate Session value is absent. Unbound/removed
/// names have no tree key; there is no second live namespace map.
#[derive(Clone, Debug)]
pub(in crate::workspace::base) enum BindingTreeValue {
    Local(RootedWorkspaceEntry),
    Global(GlobalBindingTarget),
}

/// Immutable AVL generation. Cloning shares one root and every retained
/// value/name owner beneath it; it does not clone a language `Value`.
#[derive(Clone, Debug, Default)]
pub(super) struct BindingTree {
    root: Link,
}

#[derive(Debug)]
pub(super) enum SortedBindingTreeError {
    Admission(HostAllocationError),
    NonAscendingNames,
    MissingInput,
}

impl From<HostAllocationError> for SortedBindingTreeError {
    fn from(value: HostAllocationError) -> Self {
        Self::Admission(value)
    }
}

impl BindingTree {
    pub(super) fn node_requested_bytes() -> usize {
        Node::requested_bytes()
    }

    pub(super) fn len(&self) -> usize {
        Node::link_len(&self.root)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    pub(super) fn get(&self, name: &str) -> Option<&BindingTreeValue> {
        let mut cursor = self.root.as_ref();
        while let Some(node) = cursor {
            let node = node.get();
            match name.cmp(node.name.as_str()) {
                std::cmp::Ordering::Less => cursor = node.left.as_ref(),
                std::cmp::Ordering::Greater => cursor = node.right.as_ref(),
                std::cmp::Ordering::Equal => return Some(&node.binding),
            }
        }
        None
    }

    pub(super) fn local(&self, name: &str) -> Option<&RootedWorkspaceEntry> {
        match self.get(name) {
            Some(BindingTreeValue::Local(entry)) => Some(entry),
            Some(BindingTreeValue::Global(_)) | None => None,
        }
    }

    pub(super) fn global_target(&self, name: &str) -> Option<&GlobalBindingTarget> {
        match self.get(name) {
            Some(BindingTreeValue::Global(target)) => Some(target),
            Some(BindingTreeValue::Local(_)) | None => None,
        }
    }

    /// Borrow the first binding after a previous name; no key or value copy.
    pub(super) fn name_after(
        &self,
        previous: Option<&str>,
    ) -> Option<(&WorkspaceName, &BindingTreeValue)> {
        let mut cursor = self.root.as_ref();
        let mut successor = None;
        while let Some(node) = cursor {
            let candidate = node.get();
            if previous.is_some_and(|name| candidate.name.as_str() <= name) {
                cursor = candidate.right.as_ref();
            } else {
                successor = Some((&candidate.name, &candidate.binding));
                cursor = candidate.left.as_ref();
            }
        }
        successor
    }

    pub(super) fn prepare_set(
        &self,
        context: Option<&RuntimeContext>,
        name: WorkspaceName,
        binding: BindingTreeValue,
    ) -> Result<Self, HostAllocationError> {
        let (root, _) = mutation::insert(&self.root, name, binding, context)?;
        Ok(Self { root })
    }

    pub(super) fn prepare_remove(
        &self,
        context: Option<&RuntimeContext>,
        name: &str,
    ) -> Result<Option<Self>, HostAllocationError> {
        let (root, removed) = mutation::remove(&self.root, name, context)?;
        Ok(removed.then_some(Self { root }))
    }

    /// One-time migration of already rooted/admitted names and entries.
    /// `len` is checked against actual input, and strict sort order is
    /// validated while building a balanced tree in O(len) time and O(log len)
    /// stack. Every node still receives its own exact current-operation claim.
    pub(super) fn prepare_sorted_exact(
        context: Option<&RuntimeContext>,
        mut bindings: impl ExactSizeIterator<Item = (WorkspaceName, BindingTreeValue)>,
    ) -> Result<Self, SortedBindingTreeError> {
        let len = bindings.len();
        let mut previous = None;
        let root = mutation::build_sorted(
            len,
            &mut bindings,
            &mut previous,
            &mut |name, value, left, right| Node::make(name, value, left, right, context),
        )?;
        if bindings.next().is_some() {
            return Err(SortedBindingTreeError::NonAscendingNames);
        }
        Ok(Self { root })
    }

    /// One-time import using an outer receipt that quotes exactly one node
    /// claim per input binding. Unlike path-copy mutation, no nested receipts
    /// or provisional generation are needed.
    pub(super) fn prepare_sorted_with_permit(
        mut bindings: impl ExactSizeIterator<Item = (WorkspaceName, BindingTreeValue)>,
        permit: &HostOutputPermit,
    ) -> Result<Self, SortedBindingTreeError> {
        let len = bindings.len();
        let mut previous = None;
        let root = mutation::build_sorted(
            len,
            &mut bindings,
            &mut previous,
            &mut |name, value, left, right| {
                Node::make_with_permit(name, value, left, right, permit)
            },
        )?;
        if bindings.next().is_some() {
            return Err(SortedBindingTreeError::NonAscendingNames);
        }
        Ok(Self { root })
    }

    pub(super) fn same_root(&self, other: &Self) -> bool {
        match (&self.root, &other.root) {
            (None, None) => true,
            (Some(left), Some(right)) => left.ptr_eq(right),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests;
