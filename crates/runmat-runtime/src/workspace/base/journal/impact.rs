//! Pre-admitted name-only history retained by the canonical workspace owner.

use std::cell::Cell;

use runmat_execution::resource::ExactShared;
use runmat_value::{ExactHostAllocationClaim, ExactHostAllocationPermit};

use crate::context::host_allocation::{HostAllocationError, HostAllocationPlan};

use super::super::{values::WorkspaceEntryName, WorkspaceRevision};

#[derive(Clone, Debug)]
pub(in crate::workspace::base) enum WorkspaceImpact {
    Name(WorkspaceEntryName),
    Clear,
}

#[derive(Debug)]
struct WorkspaceImpactNodePayload {
    revision: Cell<WorkspaceRevision>,
    impact: WorkspaceImpact,
    previous: Option<WorkspaceImpactNode>,
    _claim: ExactHostAllocationClaim,
}

#[derive(Clone, Debug)]
struct WorkspaceImpactNode(ExactShared<WorkspaceImpactNodePayload>);

#[derive(Clone, Debug, Default)]
pub(in crate::workspace::base) struct PreparedWorkspaceImpacts {
    latest: Option<WorkspaceImpactNode>,
    len: usize,
}

impl PreparedWorkspaceImpacts {
    pub(in crate::workspace::base) const fn new() -> Self {
        Self {
            latest: None,
            len: 0,
        }
    }

    pub(in crate::workspace::base) fn retained_bytes() -> Result<u64, HostAllocationError> {
        HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(ExactShared::<WorkspaceImpactNodePayload>::requested_bytes())
                .map_err(|_| HostAllocationError::Overflow)?,
            1,
        )
    }

    pub(in crate::workspace::base) fn retained_bytes_for_name(
        &self,
        name: &str,
    ) -> Result<u64, HostAllocationError> {
        if self.contains_name(name) {
            Ok(0)
        } else {
            Self::retained_bytes()
        }
    }

    pub(in crate::workspace::base) fn retained_bytes_for_clear(
        &self,
    ) -> Result<u64, HostAllocationError> {
        if self.has_newest_clear() {
            Ok(0)
        } else {
            Self::retained_bytes()
        }
    }

    pub(in crate::workspace::base) fn push(
        &self,
        impact: WorkspaceImpact,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, HostAllocationError> {
        if self.contains_equivalent(&impact) {
            return Ok(self.clone());
        }
        let claim = permit
            .reserve_exact_claim(ExactShared::<WorkspaceImpactNodePayload>::requested_bytes())?;
        let payload = WorkspaceImpactNodePayload {
            revision: Cell::new(WorkspaceRevision::INITIAL),
            impact,
            previous: self.latest.clone(),
            _claim: claim,
        };
        let latest = ExactShared::try_new(payload)
            .map(WorkspaceImpactNode)
            .map_err(|_| {
                HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                )
            })?;
        Ok(Self {
            latest: Some(latest),
            len: self
                .len
                .checked_add(1)
                .ok_or(HostAllocationError::Overflow)?,
        })
    }

    fn contains_equivalent(&self, candidate: &WorkspaceImpact) -> bool {
        match candidate {
            WorkspaceImpact::Name(name) => self.contains_name(name.as_str()),
            WorkspaceImpact::Clear => self.has_newest_clear(),
        }
    }

    fn contains_name(&self, candidate: &str) -> bool {
        let mut node = self.latest.as_ref();
        while let Some(current) = node {
            let payload = current.0.get();
            match &payload.impact {
                WorkspaceImpact::Name(existing) if existing.as_str() == candidate => return true,
                WorkspaceImpact::Clear => return false,
                _ => {}
            }
            node = payload.previous.as_ref();
        }
        false
    }

    fn has_newest_clear(&self) -> bool {
        self.latest
            .as_ref()
            .is_some_and(|latest| matches!(&latest.0.get().impact, WorkspaceImpact::Clear))
    }

    pub(in crate::workspace::base) fn is_empty(&self) -> bool {
        self.latest.is_none()
    }

    pub(in crate::workspace::base) fn len(&self) -> usize {
        self.len
    }

    pub(in crate::workspace::base) fn stamp(&self, revision: WorkspaceRevision) {
        let mut node = self.latest.as_ref();
        while let Some(current) = node {
            let payload = current.0.get();
            payload.revision.set(revision);
            node = payload.previous.as_ref();
        }
    }

    pub(in crate::workspace::base) fn attach_older(&mut self, older: Self) {
        let Some(mut current) = self.latest.as_mut() else {
            *self = older;
            return;
        };
        loop {
            debug_assert!(current.0.is_unique(), "pending history must remain unique");
            let payload = current.0.get_mut_unique();
            if payload.previous.is_none() {
                payload.previous = older.latest;
                self.len += older.len;
                return;
            }
            current = payload
                .previous
                .as_mut()
                .expect("previous history node was checked above");
        }
    }

    pub(in crate::workspace::base) fn truncate(
        &mut self,
        limit: usize,
    ) -> Option<WorkspaceRevision> {
        if self.len <= limit {
            return None;
        }
        if limit == 0 {
            let retired = self.latest.take();
            let floor = retired.as_ref().map(|node| node.0.get().revision.get());
            self.len = 0;
            drop(retired);
            return floor;
        }
        let mut current = self.latest.as_mut()?;
        for _ in 1..limit {
            current = current.0.get_mut_unique().previous.as_mut()?;
        }
        let retired = current.0.get_mut_unique().previous.take();
        let floor = retired.as_ref().map(|node| node.0.get().revision.get());
        self.len = limit;
        drop(retired);
        floor
    }

    pub(in crate::workspace::base) fn iter_newest(&self) -> WorkspaceImpactIter<'_> {
        WorkspaceImpactIter {
            next: self.latest.as_ref(),
        }
    }
}

pub(in crate::workspace::base) struct WorkspaceImpactIter<'a> {
    next: Option<&'a WorkspaceImpactNode>,
}

impl<'a> Iterator for WorkspaceImpactIter<'a> {
    type Item = (WorkspaceRevision, &'a WorkspaceImpact);

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.next?;
        let payload = node.0.get();
        self.next = payload.previous.as_ref();
        Some((payload.revision.get(), &payload.impact))
    }
}

impl Drop for WorkspaceImpactNodePayload {
    fn drop(&mut self) {
        let mut previous = self.previous.take();
        while let Some(mut node) = previous {
            if !node.0.is_unique() {
                break;
            }
            previous = node.0.get_mut_unique().previous.take();
            drop(node);
        }
    }
}
