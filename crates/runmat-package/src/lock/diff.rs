use super::PackageLock;
use crate::ContentDigest;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LockDiff {
    pub added: BTreeSet<ContentDigest>,
    pub removed: BTreeSet<ContentDigest>,
    pub selection_changed: bool,
    pub root_changed: bool,
}

impl LockDiff {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty()
            && self.removed.is_empty()
            && !self.selection_changed
            && !self.root_changed
    }
}

pub fn diff_locks(before: &PackageLock, after: &PackageLock) -> LockDiff {
    let before_instances = before
        .packages
        .iter()
        .map(|package| package.instance.identity_digest.clone())
        .collect::<BTreeSet<_>>();
    let after_instances = after
        .packages
        .iter()
        .map(|package| package.instance.identity_digest.clone())
        .collect::<BTreeSet<_>>();
    LockDiff {
        added: after_instances
            .difference(&before_instances)
            .cloned()
            .collect(),
        removed: before_instances
            .difference(&after_instances)
            .cloned()
            .collect(),
        selection_changed: before.selection != after.selection,
        root_changed: before.root != after.root,
    }
}
