//! Exact ordered mutations layered over an immutable workspace generation.

use crate::context::host_allocation::{HostAllocationError, HostAllocationPlan, HostOutputPermit};
use runmat_execution::resource::ExactShared;
use runmat_value::{ExactHostAllocationClaim, ExactHostAllocationPermit, Value};

use super::super::prepared_write::PreparedWorkspaceWrite;
use super::super::values::RootedWorkspaceEntry;
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

    fn assigned_name(&self) -> Option<&str> {
        match self {
            Self::Assign(write) => Some(write.name().as_str()),
            Self::OwnedAssign { name, .. } => Some(name.as_str()),
            Self::Namespace { .. } | Self::Clear => None,
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

    pub(in crate::workspace::base) fn clear_retained_bytes() -> Result<u64, HostAllocationError> {
        Self::node_retained_bytes()
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

    /// Once the old projection owner has retired, fold an uninterrupted
    /// repeated owned store into its unique predecessor. A shared suffix is
    /// retained verbatim for already-captured readers.
    pub(super) fn coalesce_latest_owned_assignment(&mut self) -> Option<Self> {
        let mut head = self.latest.take()?;
        if !head.0.is_unique() {
            self.latest = Some(head);
            return None;
        }
        if !matches!(
            head.0.get().action,
            WorkspaceOverlayAction::OwnedAssign { .. }
        ) {
            self.latest = Some(head);
            return None;
        }
        let replaced = {
            let payload = head.0.get_mut_unique();
            let WorkspaceOverlayAction::OwnedAssign { name, entry } = &mut payload.action else {
                unreachable!("checked owned assignment")
            };
            let mut current = payload.previous.as_mut();
            let mut replaced = false;
            while let Some(node) = current {
                if !node.0.is_unique() {
                    break;
                }
                let previous = node.0.get_mut_unique();
                match &mut previous.action {
                    WorkspaceOverlayAction::OwnedAssign {
                        name: old_name,
                        entry: retained,
                    } if old_name.as_str() == name.as_str() => {
                        std::mem::swap(retained, entry);
                        replaced = true;
                        break;
                    }
                    WorkspaceOverlayAction::Assign(write)
                        if write.name().as_str() == name.as_str() =>
                    {
                        break;
                    }
                    WorkspaceOverlayAction::Namespace { name: changed, .. }
                        if changed.as_str() == name.as_str() =>
                    {
                        break;
                    }
                    WorkspaceOverlayAction::Clear => break,
                    _ => current = previous.previous.as_mut(),
                }
            }
            replaced
        };
        if replaced {
            self.latest = head.0.get_mut_unique().previous.take();
            Some(Self { latest: Some(head) })
        } else {
            self.latest = Some(head);
            None
        }
    }

    pub(super) fn stage_clear(
        &mut self,
        permit: &HostOutputPermit,
    ) -> Result<(), HostAllocationError> {
        self.push(WorkspaceOverlayAction::Clear, permit)
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

    pub(super) fn attach_canonical_tail(&mut self, canonical: &Self) {
        let mut tail = canonical.latest.clone();
        if let Some(previous) = tail.as_ref() {
            if let Some(name) = previous.0.get().action.assigned_name() {
                if !matches!(self.lookup(name), WorkspaceOverlayLookup::Unmentioned) {
                    // The pending suffix has already superseded this one
                    // committed assignment. An old snapshot owns its node;
                    // the new chain can share its predecessor directly.
                    tail = previous.0.get().previous.clone();
                }
            }
        }
        let Some(node) = self.latest.as_mut() else {
            self.latest = tail;
            return;
        };
        attach_tail(node, tail);
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

fn attach_tail(node: &mut WorkspaceOverlayNode, canonical: Option<WorkspaceOverlayNode>) {
    let mut current = node;
    loop {
        debug_assert!(
            current.0.is_unique(),
            "unpublished overlay suffix must be unique"
        );
        let payload = current.0.get_mut_unique();
        if payload.previous.is_none() {
            payload.previous = canonical;
            return;
        }
        current = payload.previous.as_mut().expect("checked predecessor");
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
