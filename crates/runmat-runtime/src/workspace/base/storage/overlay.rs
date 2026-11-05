//! Exact ordered mutations layered over an immutable workspace generation.

use crate::context::host_allocation::{HostAllocationError, HostAllocationPlan, HostOutputPermit};
use runmat_execution::resource::ExactShared;
use runmat_value::{ExactHostAllocationClaim, ExactHostAllocationPermit, Value};

use super::super::prepared_write::PreparedWorkspaceWrite;
use super::super::values::RootedWorkspaceEntry;
use super::binding_tree::{BindingTree, BindingTreeValue};
use crate::workspace::retained::{WorkspaceName, WorkspaceRetainedValue};
use crate::workspace::{GlobalBindingTarget, WorkspaceSlotReadLease};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::workspace::base) enum WorkspaceNamespaceAction {
    Remove,
    BindGlobal,
    UnbindGlobal,
}

#[derive(Clone, Debug)]
enum WorkspaceOverlayAction {
    Assign(PreparedWorkspaceWrite),
    OwnedAssign {
        name: WorkspaceName,
        entry: RootedWorkspaceEntry,
    },
    Namespace {
        name: WorkspaceName,
        action: WorkspaceNamespaceAction,
        target: Option<GlobalBindingTarget>,
    },
    Clear,
}

impl WorkspaceOverlayAction {
    fn name(&self) -> Option<&WorkspaceName> {
        match self {
            Self::Assign(write) => Some(write.name()),
            Self::OwnedAssign { name, .. } => Some(name),
            Self::Namespace { name, .. } => Some(name),
            Self::Clear => None,
        }
    }
}

#[derive(Debug)]
struct WorkspaceOverlayNodePayload {
    action: WorkspaceOverlayAction,
    previous: Option<WorkspaceOverlayNode>,
    _claim: Option<ExactHostAllocationClaim>,
}

impl Drop for WorkspaceOverlayNodePayload {
    fn drop(&mut self) {
        // The final owner of a shared suffix can expose an arbitrarily long
        // uniquely owned tail. Detach each predecessor before releasing its
        // node so destruction uses constant call-stack space.
        let mut previous = self.previous.take();
        while let Some(mut node) = previous {
            if !node.0.is_unique() {
                break;
            }
            previous = node.0.get_mut_unique().previous.take();
        }
    }
}

#[derive(Clone, Debug)]
struct WorkspaceOverlayNode(ExactShared<WorkspaceOverlayNodePayload>);

/// A complete, admitted node with no predecessor yet. It is allocated before
/// the live projection can change, then either links a shared suffix or
/// exchanges its action with a uniquely owned prior assignment.
pub(super) struct PreparedOverlayAssignment(WorkspaceOverlayNode);

impl PreparedOverlayAssignment {
    pub(super) fn name(&self) -> &WorkspaceName {
        let WorkspaceOverlayAction::Assign(write) = &self.0 .0.get().action else {
            unreachable!("prepared overlay assignment")
        };
        write.name()
    }
}

#[derive(Clone, Debug, Default)]
pub(in crate::workspace::base) struct WorkspaceOverlay {
    latest: Option<WorkspaceOverlayNode>,
}

pub(super) enum WorkspaceOverlayLookup<'a> {
    Unmentioned,
    PreparedValue(&'a WorkspaceRetainedValue),
    OwnedValue(&'a RootedWorkspaceEntry),
    Absent,
}

impl WorkspaceOverlay {
    /// Explicitly unbounded direct-owner entry to the same path-copy replay.
    pub(super) fn prepare_unbounded_tree(
        &self,
        tree: BindingTree,
    ) -> Result<BindingTree, HostAllocationError> {
        self.prepare_tree(None, tree)
    }

    /// Replay only touched bindings. Scratch and path copies use the actual
    /// operation owner; retained values and captured trees are shared.
    pub(super) fn prepare_tree(
        &self,
        context: Option<&crate::context::RuntimeContext>,
        mut tree: BindingTree,
    ) -> Result<BindingTree, HostAllocationError> {
        let mut count = 0usize;
        let mut next = self.latest.as_ref();
        while let Some(node) = next {
            let payload = node.0.get();
            if matches!(payload.action, WorkspaceOverlayAction::Clear) {
                tree = BindingTree::default();
                break;
            }
            count = count.checked_add(1).ok_or(HostAllocationError::Overflow)?;
            next = payload.previous.as_ref();
        }
        if count == 0 {
            return Ok(tree);
        }
        let bytes = std::alloc::Layout::array::<&WorkspaceOverlayAction>(count)
            .map_err(|_| HostAllocationError::Overflow)?
            .size();
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(bytes).map_err(|_| HostAllocationError::Overflow)?,
            usize::from(count != 0),
        )?;
        let receipt = crate::context::host_allocation::HostAllocationReceipt::admit(
            context,
            HostAllocationPlan::new(0, retained)?,
        )?;
        let claim = receipt
            .as_ref()
            .map(|receipt| {
                receipt
                    .output_permit(context.expect("managed replay has an operation owner"))?
                    .reserve_exact_claim(bytes)
                    .map_err(HostAllocationError::from)
            })
            .transpose()?;
        // Fields drop in order: physical scratch before its funding claim.
        struct Actions<'a> {
            values: Vec<&'a WorkspaceOverlayAction>,
            _claim: Option<ExactHostAllocationClaim>,
        }
        let mut actions = Actions {
            values: runmat_value::exact_capacity_vec(count)
                .map_err(HostAllocationError::Numeric)?,
            _claim: claim,
        };
        let mut next = self.latest.as_ref();
        for _ in 0..count {
            let payload = next.expect("counted immutable overlay actions").0.get();
            actions.values.push(&payload.action);
            next = payload.previous.as_ref();
        }
        for action in actions.values.iter().rev().copied() {
            match action {
                WorkspaceOverlayAction::Assign(write) => {
                    tree = tree.prepare_set(
                        context,
                        write.name().clone(),
                        BindingTreeValue::Local(RootedWorkspaceEntry::from_prepared(write)),
                    )?;
                }
                WorkspaceOverlayAction::OwnedAssign { name, entry } => {
                    tree = tree.prepare_set(
                        context,
                        name.clone(),
                        BindingTreeValue::Local(entry.clone()),
                    )?;
                }
                WorkspaceOverlayAction::Namespace {
                    name,
                    action,
                    target,
                } => match action {
                    WorkspaceNamespaceAction::BindGlobal => {
                        let target = target.as_ref().ok_or(HostAllocationError::StaleAuthority)?;
                        tree = tree.prepare_set(
                            context,
                            name.clone(),
                            BindingTreeValue::Global(target.clone()),
                        )?;
                    }
                    WorkspaceNamespaceAction::Remove
                        if tree.global_target(name.as_str()).is_some() => {}
                    WorkspaceNamespaceAction::Remove | WorkspaceNamespaceAction::UnbindGlobal => {
                        tree = tree.prepare_remove(context, name.as_str())?.unwrap_or(tree);
                    }
                },
                WorkspaceOverlayAction::Clear => unreachable!("clear bounds the replay suffix"),
            }
        }
        Ok(tree)
    }

    pub(in crate::workspace::base) fn retained_bytes(
        name: &str,
        value: &Value,
    ) -> Result<Option<u64>, HostAllocationError> {
        let Some(write) = PreparedWorkspaceWrite::retained_bytes(name, value)? else {
            return Ok(None);
        };
        write
            .checked_add(Self::node_retained_bytes()?)
            .map(Some)
            .ok_or(HostAllocationError::Overflow)
    }

    pub(in crate::workspace::base) fn retained_generation_bytes(
        name: &str,
        source: &WorkspaceSlotReadLease,
    ) -> Result<Option<u64>, HostAllocationError> {
        let Some(write) = PreparedWorkspaceWrite::retained_generation_bytes(name, source)? else {
            return Ok(None);
        };
        write
            .checked_add(Self::node_retained_bytes()?)
            .map(Some)
            .ok_or(HostAllocationError::Overflow)
    }

    pub(in crate::workspace::base) fn namespace_retained_bytes(
        name: &str,
    ) -> Result<u64, HostAllocationError> {
        let raw = WorkspaceName::requested_bytes(name)?;
        let name = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(raw).map_err(|_| HostAllocationError::Overflow)?,
            1,
        )?;
        name.checked_add(Self::node_retained_bytes()?)
            .ok_or(HostAllocationError::Overflow)
    }

    fn node_retained_bytes() -> Result<u64, HostAllocationError> {
        HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(ExactShared::<WorkspaceOverlayNodePayload>::requested_bytes())
                .map_err(|_| HostAllocationError::Overflow)?,
            1,
        )
    }

    pub(super) fn stage_prepared(
        &mut self,
        write: PreparedWorkspaceWrite,
        permit: &HostOutputPermit,
    ) -> Result<(), HostAllocationError> {
        if replace_unique_assignment(&mut self.latest, &write) {
            return Ok(());
        }
        self.push(WorkspaceOverlayAction::Assign(write), permit)
    }

    pub(super) fn prepare_assignment(
        write: PreparedWorkspaceWrite,
        permit: &HostOutputPermit,
    ) -> Result<PreparedOverlayAssignment, HostAllocationError> {
        let claim = permit
            .reserve_exact_claim(ExactShared::<WorkspaceOverlayNodePayload>::requested_bytes())?;
        let payload = WorkspaceOverlayNodePayload {
            action: WorkspaceOverlayAction::Assign(write),
            previous: None,
            _claim: Some(claim),
        };
        ExactShared::try_new(payload)
            .map(|owner| PreparedOverlayAssignment(WorkspaceOverlayNode(owner)))
            .map_err(|_| {
                HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                )
            })
    }

    /// Install a fully prepared node without allocation or Value destruction.
    /// A unique old write is moved into the unused prepared node and returned
    /// for retirement after the service/observer borrow has ended.
    pub(super) fn install_prepared_assignment(
        &mut self,
        mut prepared: PreparedOverlayAssignment,
    ) -> Option<Self> {
        if replace_unique_assignment_with_retirement(&mut self.latest, &mut prepared.0) {
            Some(Self {
                latest: Some(prepared.0),
            })
        } else {
            prepared.0 .0.get_mut_unique().previous = self.latest.take();
            self.latest = Some(prepared.0);
            None
        }
    }

    pub(super) fn stage_namespace(
        &mut self,
        name: WorkspaceName,
        action: WorkspaceNamespaceAction,
        target: Option<GlobalBindingTarget>,
        permit: &HostOutputPermit,
    ) -> Result<(), HostAllocationError> {
        self.push(
            WorkspaceOverlayAction::Namespace {
                name,
                action,
                target,
            },
            permit,
        )
    }

    pub(super) fn stage_namespace_unbounded(
        &mut self,
        name: WorkspaceName,
        action: WorkspaceNamespaceAction,
        target: Option<GlobalBindingTarget>,
    ) {
        self.latest = Some(WorkspaceOverlayNode(ExactShared::new(
            WorkspaceOverlayNodePayload {
                action: WorkspaceOverlayAction::Namespace {
                    name,
                    action,
                    target,
                },
                previous: self.latest.take(),
                _claim: None,
            },
        )));
    }

    /// Prepare an explicitly unbounded action off-state, before a prospective
    /// observer or any visible projection mutation. Installing the prepared
    /// overlay only links already-owned nodes.
    pub(super) fn stage_owned_unbounded(
        &mut self,
        name: WorkspaceName,
        entry: RootedWorkspaceEntry,
    ) {
        self.latest = Some(WorkspaceOverlayNode(ExactShared::new(
            WorkspaceOverlayNodePayload {
                action: WorkspaceOverlayAction::OwnedAssign { name, entry },
                previous: self.latest.take(),
                _claim: None,
            },
        )));
    }

    pub(super) fn stage_clear_unbounded(&mut self) {
        self.latest = Some(WorkspaceOverlayNode(ExactShared::new(
            WorkspaceOverlayNodePayload {
                action: WorkspaceOverlayAction::Clear,
                previous: self.latest.take(),
                _claim: None,
            },
        )));
    }

    fn push(
        &mut self,
        action: WorkspaceOverlayAction,
        permit: &HostOutputPermit,
    ) -> Result<(), HostAllocationError> {
        let claim = permit
            .reserve_exact_claim(ExactShared::<WorkspaceOverlayNodePayload>::requested_bytes())?;
        let payload = WorkspaceOverlayNodePayload {
            action,
            previous: self.latest.take(),
            _claim: Some(claim),
        };
        match ExactShared::try_new(payload) {
            Ok(owner) => {
                self.latest = Some(WorkspaceOverlayNode(owner));
                Ok(())
            }
            Err(mut payload) => {
                self.latest = payload.previous.take();
                Err(HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                ))
            }
        }
    }

    pub(super) fn lookup(&self, name: &str) -> WorkspaceOverlayLookup<'_> {
        let mut node = self.latest.as_ref();
        while let Some(current) = node {
            let payload = current.0.get();
            match &payload.action {
                WorkspaceOverlayAction::Assign(write) if write.name().as_str() == name => {
                    return WorkspaceOverlayLookup::PreparedValue(write.value());
                }
                WorkspaceOverlayAction::OwnedAssign {
                    name: changed,
                    entry,
                } if changed.as_str() == name => {
                    return WorkspaceOverlayLookup::OwnedValue(entry);
                }
                WorkspaceOverlayAction::Namespace { name: changed, .. }
                    if changed.as_str() == name =>
                {
                    return WorkspaceOverlayLookup::Absent
                }
                WorkspaceOverlayAction::Clear => return WorkspaceOverlayLookup::Absent,
                _ => {}
            }
            node = payload.previous.as_ref();
        }
        WorkspaceOverlayLookup::Unmentioned
    }

    pub(super) fn global_target(&self, name: &str) -> Option<Option<&GlobalBindingTarget>> {
        let mut node = self.latest.as_ref();
        while let Some(current) = node {
            let payload = current.0.get();
            match &payload.action {
                WorkspaceOverlayAction::Namespace {
                    name: changed,
                    action,
                    target,
                } if changed.as_str() == name => match action {
                    WorkspaceNamespaceAction::BindGlobal => return Some(target.as_ref()),
                    WorkspaceNamespaceAction::UnbindGlobal => return Some(None),
                    WorkspaceNamespaceAction::Remove => {}
                },
                WorkspaceOverlayAction::Assign(write) if write.name().as_str() == name => {
                    return Some(None);
                }
                WorkspaceOverlayAction::OwnedAssign { name: changed, .. }
                    if changed.as_str() == name =>
                {
                    return Some(None);
                }
                WorkspaceOverlayAction::Clear => return Some(None),
                _ => {}
            }
            node = payload.previous.as_ref();
        }
        None
    }

    pub(super) fn global(&self, name: &str) -> Option<bool> {
        self.global_target(name).map(|target| target.is_some())
    }

    pub(super) fn is_empty(&self) -> bool {
        self.latest.is_none()
    }

    pub(super) fn names(&self) -> WorkspaceOverlayNames<'_> {
        WorkspaceOverlayNames {
            next: self.latest.as_ref(),
        }
    }
}

pub(super) struct WorkspaceOverlayNames<'a> {
    next: Option<&'a WorkspaceOverlayNode>,
}

impl<'a> Iterator for WorkspaceOverlayNames<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let node = self.next?;
            let payload = node.0.get();
            self.next = payload.previous.as_ref();
            if let Some(name) = payload.action.name() {
                return Some(name.as_str());
            }
        }
    }
}

fn replace_unique_assignment(
    node: &mut Option<WorkspaceOverlayNode>,
    replacement: &PreparedWorkspaceWrite,
) -> bool {
    let Some(WorkspaceOverlayAction::Assign(write)) =
        unique_assignment_action(node, replacement.name().as_str())
    else {
        return false;
    };
    *write = replacement.clone();
    true
}

fn replace_unique_assignment_with_retirement(
    node: &mut Option<WorkspaceOverlayNode>,
    retirement: &mut WorkspaceOverlayNode,
) -> bool {
    let WorkspaceOverlayAction::Assign(write) = &retirement.0.get().action else {
        unreachable!("prepared overlay assignment")
    };
    let name = write.name().clone();
    let Some(action) = unique_assignment_action(node, name.as_str()) else {
        return false;
    };
    std::mem::swap(action, &mut retirement.0.get_mut_unique().action);
    true
}

fn unique_assignment_action<'a>(
    node: &'a mut Option<WorkspaceOverlayNode>,
    name: &str,
) -> Option<&'a mut WorkspaceOverlayAction> {
    let mut current = node.as_mut();
    while let Some(node) = current {
        if !node.0.is_unique() {
            return None;
        }
        let payload = node.0.get_mut_unique();
        match &payload.action {
            WorkspaceOverlayAction::Assign(write) if write.name().as_str() == name => {
                return Some(&mut payload.action);
            }
            WorkspaceOverlayAction::Namespace { name: changed, .. } if changed.as_str() == name => {
                return None;
            }
            WorkspaceOverlayAction::Clear => return None,
            WorkspaceOverlayAction::OwnedAssign { name: changed, .. }
                if changed.as_str() == name =>
            {
                return None;
            }
            _ => current = payload.previous.as_mut(),
        }
    }
    None
}
