//! Borrowed origin facts from current or prospective canonical workspace state.

use super::{WorkspaceChangeData, WorkspacePublicationOrigin};
use super::{WorkspacePublicationMutation, WorkspacePublicationStamp};
use crate::workspace::base::projection::PreparedProjectionPublicationView;
use crate::workspace::base::{
    BaseWorkspaceProjection, WorkspaceChange, WorkspaceChangeKind, WorkspaceDelta,
};
use crate::workspace::GlobalBindingTarget;

#[derive(Clone, Copy)]
enum PublicationSource<'a> {
    Current {
        changes: &'a WorkspaceDelta,
        projection: &'a BaseWorkspaceProjection,
    },
    Prepared(&'a PreparedProjectionPublicationView<'a>),
}

impl<'a> PublicationSource<'a> {
    fn change(self, index: usize) -> Option<&'a WorkspaceChange> {
        match self {
            Self::Current { changes, .. } => changes.changes().get(index),
            Self::Prepared(view) => view.change(index),
        }
    }

    fn global_target(self, name: &str) -> Option<&'a GlobalBindingTarget> {
        match self {
            Self::Current { projection, .. } => projection.global_target(name),
            Self::Prepared(view) => view.global_target(name),
        }
    }

    fn contains_value(self, name: &str) -> bool {
        match self {
            Self::Current { projection, .. } => projection.lookup(name).is_some(),
            Self::Prepared(view) => view.lookup(name).is_some(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct WorkspacePublicationBatch<'a> {
    source: PublicationSource<'a>,
    indices: &'a [usize],
    cleared: bool,
    stamp: Option<WorkspacePublicationStamp>,
}

impl<'a> WorkspacePublicationBatch<'a> {
    pub(in crate::workspace::base::projection_service) fn for_changes(
        changes: &'a WorkspaceDelta,
        projection: &'a BaseWorkspaceProjection,
        indices: &'a [usize],
    ) -> Self {
        Self {
            source: PublicationSource::Current {
                changes,
                projection,
            },
            indices,
            stamp: None,
            cleared: changes
                .changes()
                .iter()
                .any(|change| change.kind() == WorkspaceChangeKind::Clear),
        }
    }

    pub(in crate::workspace::base::projection_service) fn prospective(
        view: &'a PreparedProjectionPublicationView<'a>,
        indices: &'a [usize],
    ) -> Self {
        Self {
            source: PublicationSource::Prepared(view),
            indices,
            stamp: None,
            cleared: view
                .changes()
                .any(|change| change.kind() == WorkspaceChangeKind::Clear),
        }
    }

    pub(in crate::workspace::base::projection_service) fn with_stamp(
        mut self,
        stamp: WorkspacePublicationStamp,
    ) -> Self {
        self.stamp = Some(stamp);
        self
    }

    pub fn publication_stamp(self) -> Option<WorkspacePublicationStamp> {
        self.stamp
    }

    pub fn is_empty(self) -> bool {
        self.iter().next().is_none()
    }

    pub fn len(self) -> usize {
        self.iter().count()
    }

    /// Iterate the canonical final effect without reconstructing names or
    /// values. Superseded source actions are skipped, and a final clear is
    /// emitted before the surviving suffix.
    pub fn iter(self) -> impl Iterator<Item = WorkspacePublicationOrigin<'a>> + 'a {
        WorkspacePublicationIter {
            batch: self,
            position: 0,
            emit_clear: self.cleared,
        }
    }
}

impl<'a> WorkspacePublicationMutation<'a> {
    pub(in crate::workspace::base::projection_service) fn prospective(
        view: &'a PreparedProjectionPublicationView<'a>,
        issuer: usize,
    ) -> Self {
        assert!(
            view.matches_base(),
            "prospective publication requires its original projection"
        );
        let change = view.mutation_change().expect("single prepared mutation");
        let source = PublicationSource::Prepared(view);
        let origin = match change.data() {
            WorkspaceChangeData::Clear => WorkspacePublicationOrigin::Cleared,
            WorkspaceChangeData::BindGlobal { target, .. } => {
                WorkspacePublicationOrigin::BindGlobal {
                    name: change.name().expect("named global binding"),
                    target,
                }
            }
            WorkspaceChangeData::UnbindGlobal { .. } => WorkspacePublicationOrigin::UnbindGlobal {
                name: change.name().expect("named global unbinding"),
            },
            WorkspaceChangeData::Remove { .. } => {
                let name = change.name().expect("named removal");
                match source.global_target(name) {
                    Some(target) => WorkspacePublicationOrigin::BindGlobal { name, target },
                    None => WorkspacePublicationOrigin::Retired { name },
                }
            }
            WorkspaceChangeData::Assign { .. } => {
                let name = change.name().expect("named assignment");
                let global = source.global_target(name);
                assert!(
                    global.is_some() || source.contains_value(name),
                    "prepared assignment retains its value or target"
                );
                match change.executor_slot() {
                    Some(slot) => WorkspacePublicationOrigin::ExecutorSlot { name, slot, global },
                    None => WorkspacePublicationOrigin::DynamicEffect { name, global },
                }
            }
        };
        Self::issued(
            WorkspacePublicationStamp::issued(
                issuer,
                view.revision(),
                view.before_generation(),
                view.generation(),
            ),
            origin,
        )
    }
}

struct WorkspacePublicationIter<'a> {
    batch: WorkspacePublicationBatch<'a>,
    position: usize,
    emit_clear: bool,
}

impl<'a> Iterator for WorkspacePublicationIter<'a> {
    type Item = WorkspacePublicationOrigin<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.emit_clear {
            self.emit_clear = false;
            return Some(WorkspacePublicationOrigin::Cleared);
        }
        loop {
            let index = *self.batch.indices.get(self.position)?;
            self.position += 1;
            let change = self
                .batch
                .source
                .change(index)
                .expect("publication change index");
            let name = change.name()?;
            match change.data() {
                WorkspaceChangeData::BindGlobal { target, .. } => {
                    return Some(WorkspacePublicationOrigin::BindGlobal { name, target });
                }
                WorkspaceChangeData::UnbindGlobal { .. } => {
                    return Some(WorkspacePublicationOrigin::UnbindGlobal { name });
                }
                WorkspaceChangeData::Assign { .. }
                | WorkspaceChangeData::Remove { .. }
                | WorkspaceChangeData::Clear => {}
            }
            let global = self.batch.source.global_target(name);
            if matches!(change.data(), WorkspaceChangeData::Remove { .. }) {
                if let Some(target) = global {
                    return Some(WorkspacePublicationOrigin::BindGlobal { name, target });
                }
            }
            if global.is_some() || self.batch.source.contains_value(name) {
                return Some(match change.executor_slot() {
                    Some(slot) => WorkspacePublicationOrigin::ExecutorSlot { name, slot, global },
                    None => WorkspacePublicationOrigin::DynamicEffect { name, global },
                });
            }
            if !self.batch.cleared {
                return Some(WorkspacePublicationOrigin::Retired { name });
            }
        }
    }
}
