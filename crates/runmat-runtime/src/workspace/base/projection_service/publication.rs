//! Binding-origin facts accompanying successful canonical workspace commits.
//!
//! Values remain owned by the projection and base workspace. These events
//! carry the final binding class and, for compiled stores, the exact executor
//! slot of the canonical mutation.

use std::alloc::Layout;

use crate::context::host_allocation::{HostAllocationError, HostAllocationPlan};
use crate::RuntimeError;
use runmat_value::{ExactHostAllocationClaim, ExactHostAllocationPermit};

use super::super::delta::WorkspaceChangeData;
use super::super::{WorkspaceChange, WorkspaceChangeKind, WorkspaceDelta};
use crate::workspace::base::projection::PreparedProjectionPublicationView;
use crate::workspace::GlobalBindingTarget;

mod observer;
mod batch;
mod mutation;
pub use batch::WorkspacePublicationBatch;
pub use mutation::{WorkspacePublicationMutation, WorkspacePublicationStamp};
pub(in crate::workspace::base::projection_service) use observer::PreparedWorkspacePublication;
pub use observer::{
    WorkspacePublicationObserver, WorkspacePublicationProtocol, WorkspacePublicationToken,
};

#[derive(Debug, Default)]
pub(super) struct PublishedWorkspaceEffect {
    cleared: bool,
    changes: PublishedWorkspaceBuffer,
    retirement: PublishedWorkspaceBuffer,
    /// Empty, preadmitted backing for the next independent canonical effect.
    /// Authentic pending owners remain in the projection through publication.
    canonical_effect: PublishedWorkspaceBuffer,
    publication_indices: PublicationIndexBuffer,
}

impl PublishedWorkspaceEffect {
    pub(super) fn can_observe(&self, pending_change_upper_bound: usize) -> bool {
        let named_capacity = match self
            .changes
            .changes
            .len()
            .checked_add(pending_change_upper_bound)
        {
            Some(capacity) => capacity,
            None => return false,
        };
        self.changes.changes.capacity() >= named_capacity
            && self.retirement.changes.capacity() >= named_capacity
            && self.canonical_effect.changes.capacity() >= pending_change_upper_bound
            && self.publication_indices.indices.capacity() >= pending_change_upper_bound
    }

    pub(super) fn retained_bytes_for_upper_bound(
        &self,
        pending_change_upper_bound: usize,
    ) -> Result<u64, HostAllocationError> {
        let named_capacity = self
            .changes
            .changes
            .len()
            .checked_add(pending_change_upper_bound)
            .ok_or(HostAllocationError::Overflow)?;
        let output_capacity = named_capacity
            .checked_add(1)
            .ok_or(HostAllocationError::Overflow)?;
        let retained = self
            .changes
            .additional_retained_bytes(output_capacity)?
            .checked_add(self.retirement.additional_retained_bytes(named_capacity)?)
            .ok_or(HostAllocationError::Overflow)?;
        let retained = retained
            .checked_add(
                self.canonical_effect
                    .additional_retained_bytes(pending_change_upper_bound)?,
            )
            .ok_or(HostAllocationError::Overflow)?;
        retained
            .checked_add(
                self.publication_indices
                    .additional_retained_bytes(pending_change_upper_bound)?,
            )
            .ok_or(HostAllocationError::Overflow)
    }

    pub(super) fn prepare_for_upper_bound(
        &self,
        pending_change_upper_bound: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<PreparedPublishedWorkspaceCapacity, HostAllocationError> {
        let named_capacity = self
            .changes
            .changes
            .len()
            .checked_add(pending_change_upper_bound)
            .ok_or(HostAllocationError::Overflow)?;
        let output_capacity = named_capacity
            .checked_add(1)
            .ok_or(HostAllocationError::Overflow)?;
        Ok(PreparedPublishedWorkspaceCapacity {
            changes: self.changes.prepare_admitted(output_capacity, permit)?,
            retirement: self.retirement.prepare_admitted(named_capacity, permit)?,
            canonical_effect: self
                .canonical_effect
                .prepare_admitted(pending_change_upper_bound, permit)?,
            publication_indices: self
                .publication_indices
                .prepare_admitted(pending_change_upper_bound, permit)?,
        })
    }

    pub(super) fn install_prepared_capacity(
        &mut self,
        prepared: PreparedPublishedWorkspaceCapacity,
    ) {
        if let Some(changes) = prepared.changes {
            self.changes.install(changes);
        }
        if let Some(retirement) = prepared.retirement {
            self.retirement.install(retirement);
        }
        if let Some(canonical_effect) = prepared.canonical_effect {
            self.canonical_effect.install(canonical_effect);
        }
        if let Some(indices) = prepared.publication_indices {
            self.publication_indices.install(indices);
        }
    }

    pub(super) fn prepare_unbounded_for_upper_bound(
        &mut self,
        pending_change_upper_bound: usize,
    ) -> Result<(), RuntimeError> {
        let named_capacity = self
            .changes
            .changes
            .len()
            .checked_add(pending_change_upper_bound)
            .ok_or_else(publication_capacity_error)?;
        let output_capacity = named_capacity
            .checked_add(1)
            .ok_or_else(publication_capacity_error)?;
        self.changes.prepare_unbounded(output_capacity)?;
        self.retirement.prepare_unbounded(named_capacity)?;
        self.canonical_effect
            .prepare_unbounded(pending_change_upper_bound)?;
        self.publication_indices
            .prepare_unbounded(pending_change_upper_bound)?;
        Ok(())
    }

    pub(super) fn prepare_publication_indices(&mut self, delta: &WorkspaceDelta) {
        self.publication_indices.prepare(delta);
    }

    pub(super) fn prepare_prospective_indices(
        &mut self,
        view: &PreparedProjectionPublicationView<'_>,
    ) {
        self.publication_indices
            .prepare_changes(view.changes().len(), |index| {
                view.change(index)
                    .expect("prospective publication change index")
            });
    }

    pub(super) fn publication_indices(&self) -> &[usize] {
        &self.publication_indices.indices
    }

    pub(super) fn finish_publication(&mut self) {
        self.publication_indices.indices.clear();
    }

    /// Move the already-funded empty backing into one independent effect.
    /// The projection retains the authentic source delta, including every
    /// overwritten Foreign lease, until both publication borrows retire.
    pub(super) fn take_canonical_effect(&mut self, source: &WorkspaceDelta) -> WorkspaceDelta {
        debug_assert!(self.canonical_effect.changes.is_empty());
        assert!(self.canonical_effect.changes.capacity() >= source.changes().len());
        let mut buffer = std::mem::take(&mut self.canonical_effect);
        buffer.changes.extend(source.changes().iter().cloned());
        match buffer.claim {
            Some(claim) => WorkspaceDelta::from_admitted_changes(buffer.changes, claim),
            None => WorkspaceDelta::from_changes(buffer.changes),
        }
    }

    pub(super) fn restore_canonical_effect(&mut self, effect: WorkspaceDelta) {
        debug_assert!(self.canonical_effect.changes.is_empty());
        let (mut changes, claim) = effect.into_effect_backing();
        changes.clear();
        self.canonical_effect = PublishedWorkspaceBuffer { changes, claim };
    }

    pub(super) fn observe(&mut self, delta: &WorkspaceDelta) -> PublishedWorkspaceBuffer {
        debug_assert!(self.retirement.changes.is_empty());
        for change in delta.changes() {
            match change.kind() {
                WorkspaceChangeKind::Clear => {
                    self.cleared = true;
                    self.retirement.changes.append(&mut self.changes.changes);
                }
                WorkspaceChangeKind::Assign | WorkspaceChangeKind::Remove => {
                    if let Some(name) = change.entry_name() {
                        match self.changes.changes.binary_search_by(|candidate| {
                            candidate
                                .entry_name()
                                .expect("published named change")
                                .cmp(name)
                        }) {
                            Ok(index) => {
                                self.retirement.changes.push(std::mem::replace(
                                    &mut self.changes.changes[index],
                                    change.clone(),
                                ));
                            }
                            Err(index) => {
                                self.changes.changes.insert(index, change.clone());
                            }
                        }
                    }
                }
            }
        }
        std::mem::take(&mut self.retirement)
    }

    pub(super) fn take(&mut self) -> WorkspaceDelta {
        let mut buffer = std::mem::take(&mut self.changes);
        if self.cleared {
            buffer.changes.insert(0, WorkspaceChange::clear());
        }
        self.cleared = false;
        match buffer.claim {
            Some(claim) => WorkspaceDelta::from_admitted_changes(buffer.changes, claim),
            None => WorkspaceDelta::from_changes(buffer.changes),
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct PublishedWorkspaceBuffer {
    // The Vec retires before the claim that funds its exact backing.
    changes: Vec<WorkspaceChange>,
    claim: Option<ExactHostAllocationClaim>,
}

pub(super) struct PreparedPublishedWorkspaceCapacity {
    changes: Option<PublishedWorkspaceBuffer>,
    retirement: Option<PublishedWorkspaceBuffer>,
    canonical_effect: Option<PublishedWorkspaceBuffer>,
    publication_indices: Option<PublicationIndexBuffer>,
}

#[derive(Debug, Default)]
struct PublicationIndexBuffer {
    // The Vec retires before the exact claim funding its backing.
    indices: Vec<usize>,
    claim: Option<ExactHostAllocationClaim>,
}

impl PublicationIndexBuffer {
    fn additional_retained_bytes(&self, capacity: usize) -> Result<u64, HostAllocationError> {
        if self.claim.is_some() && self.indices.capacity() >= capacity {
            return Ok(0);
        }
        let raw = Layout::array::<usize>(capacity)
            .map_err(|_| HostAllocationError::Overflow)?
            .size();
        HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(raw).map_err(|_| HostAllocationError::Overflow)?,
            1,
        )
    }

    fn prepare_admitted(
        &self,
        capacity: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Option<Self>, HostAllocationError> {
        if self.claim.is_some() && self.indices.capacity() >= capacity {
            return Ok(None);
        }
        let raw = Layout::array::<usize>(capacity)
            .map_err(|_| HostAllocationError::Overflow)?
            .size();
        let claim = permit.reserve_exact_claim(raw)?;
        Ok(Some(Self {
            indices: runmat_value::exact_capacity_vec(capacity)?,
            claim: Some(claim),
        }))
    }

    fn install(&mut self, replacement: Self) {
        debug_assert!(self.indices.is_empty());
        let retired = std::mem::replace(self, replacement);
        drop(retired);
    }

    fn prepare_unbounded(&mut self, capacity: usize) -> Result<(), RuntimeError> {
        if self.claim.is_some() {
            let mut replacement = Vec::new();
            replacement
                .try_reserve(capacity)
                .map_err(|_| publication_capacity_error())?;
            let retired = std::mem::replace(
                self,
                Self {
                    indices: replacement,
                    claim: None,
                },
            );
            drop(retired);
        } else {
            self.indices
                .try_reserve(capacity.saturating_sub(self.indices.len()))
                .map_err(|_| publication_capacity_error())?;
        }
        Ok(())
    }

    fn prepare(&mut self, delta: &WorkspaceDelta) {
        self.prepare_changes(delta.changes().len(), |index| &delta.changes()[index]);
    }

    fn prepare_changes<'a>(&mut self, len: usize, change: impl Fn(usize) -> &'a WorkspaceChange) {
        self.indices.clear();
        let start = (0..len)
            .rposition(|index| change(index).kind() == WorkspaceChangeKind::Clear)
            .map_or(0, |index| index + 1);
        self.indices
            .extend((start..len).filter(|index| change(*index).name().is_some()));
        self.indices.sort_unstable_by(|left, right| {
            let left_name = change(*left).name().expect("named publication");
            let right_name = change(*right).name().expect("named publication");
            left_name.cmp(right_name).then_with(|| right.cmp(left))
        });
        let mut write = 0;
        for read in 0..self.indices.len() {
            let index = self.indices[read];
            let unique =
                write == 0 || change(self.indices[write - 1]).name() != change(index).name();
            if unique {
                self.indices[write] = index;
                write += 1;
            }
        }
        self.indices.truncate(write);
    }
}

impl PublishedWorkspaceBuffer {
    fn additional_retained_bytes(&self, capacity: usize) -> Result<u64, HostAllocationError> {
        if self.claim.is_some() && self.changes.capacity() >= capacity {
            return Ok(0);
        }
        let raw = Layout::array::<WorkspaceChange>(capacity)
            .map_err(|_| HostAllocationError::Overflow)?
            .size();
        HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(raw).map_err(|_| HostAllocationError::Overflow)?,
            1,
        )
    }

    fn prepare_admitted(
        &self,
        capacity: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Option<Self>, HostAllocationError> {
        if self.claim.is_some() && self.changes.capacity() >= capacity {
            return Ok(None);
        }
        let capacity = capacity.max(self.changes.len());
        let raw = Layout::array::<WorkspaceChange>(capacity)
            .map_err(|_| HostAllocationError::Overflow)?
            .size();
        let claim = permit.reserve_exact_claim(raw)?;
        let changes = runmat_value::exact_capacity_vec(capacity)?;
        Ok(Some(Self {
            changes,
            claim: Some(claim),
        }))
    }

    fn install(&mut self, mut replacement: Self) {
        debug_assert!(replacement.changes.capacity() >= self.changes.len());
        replacement.changes.append(&mut self.changes);
        let retired = std::mem::replace(self, replacement);
        drop(retired);
    }

    fn prepare_unbounded(&mut self, capacity: usize) -> Result<(), RuntimeError> {
        if self.claim.is_none() {
            self.changes
                .try_reserve(capacity.saturating_sub(self.changes.len()))
                .map_err(|_| publication_capacity_error())?;
            return Ok(());
        }
        let mut replacement = Vec::new();
        replacement
            .try_reserve(capacity.max(self.changes.len()))
            .map_err(|_| publication_capacity_error())?;
        replacement.append(&mut self.changes);
        let retired = std::mem::replace(
            self,
            Self {
                changes: replacement,
                claim: None,
            },
        );
        drop(retired);
        Ok(())
    }
}

fn publication_capacity_error() -> RuntimeError {
    crate::build_runtime_error("workspace publication effect allocation failed")
        .with_identifier("RunMat:Workspace:PublicationAllocation")
        .build()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkspacePublicationOrigin<'a> {
    ExecutorSlot {
        name: &'a str,
        slot: usize,
        global: Option<&'a GlobalBindingTarget>,
    },
    DynamicEffect {
        name: &'a str,
        global: Option<&'a GlobalBindingTarget>,
    },
    BindGlobal {
        name: &'a str,
        target: &'a GlobalBindingTarget,
    },
    UnbindGlobal {
        name: &'a str,
    },
    Retired {
        name: &'a str,
    },
    Cleared,
}

impl<'a> WorkspacePublicationOrigin<'a> {
    /// A clear applies to the namespace, not to a fabricated empty name.
    pub fn name(self) -> Option<&'a str> {
        match self {
            Self::ExecutorSlot { name, .. }
            | Self::DynamicEffect { name, .. }
            | Self::BindGlobal { name, .. }
            | Self::UnbindGlobal { name }
            | Self::Retired { name } => Some(name),
            Self::Cleared => None,
        }
    }
}

#[cfg(test)]
mod tests;
