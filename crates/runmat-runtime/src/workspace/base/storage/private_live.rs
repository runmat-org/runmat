//! Private executor workspace: one persistent namespace root and one hot
//! binding. A repeated write to the hot spelling replaces only that slot;
//! changing spellings or sealing a capture path-copies the tree offstate.

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
    HostOutputPermit,
};
use crate::context::RuntimeContext;
use crate::workspace::retained::{
    is_allocation_free_workspace_alias, WorkspaceName, WorkspaceRetainedValue,
};
use crate::workspace::{GlobalBindingTarget, LogicalWorkspaceGeneration, WorkspaceSlotReadLease};
use runmat_value::Value;

use super::super::revision::WorkspaceRevision;
use super::super::values::RootedWorkspaceEntry;
use super::binding_tree::{BindingTree, BindingTreeValue};
use super::{
    claimed_name_bytes, GenerationSortScratch, ManagedWorkspaceStorage, WorkspaceOverlay,
    WorkspaceTableGeneration,
};

#[derive(Clone, Debug)]
struct PrivateHotBinding {
    name: WorkspaceName,
    // None is a real removal tombstone until the hot slot is sealed.
    binding: Option<BindingTreeValue>,
}

/// The only live private namespace state. No BTreeMap or second Value owner
/// is maintained beside this tree and its single hot override.
#[derive(Clone, Debug)]
pub(in crate::workspace::base) struct PrivateLiveStorage {
    tree: BindingTree,
    hot: Option<PrivateHotBinding>,
    revision: WorkspaceRevision,
    epoch: u64,
}

/// Offstate physical transition. The caller validates its issuer/operation
/// authority before installation, then installs matching clones into the
/// canonical owner and projection under their mutation phases.
#[derive(Clone, Debug)]
pub(in crate::workspace::base) struct PreparedPrivateTransition {
    expected_tree: BindingTree,
    expected_revision: WorkspaceRevision,
    expected_epoch: u64,
    next_tree: BindingTree,
    next_hot: Option<PrivateHotBinding>,
    next_revision: WorkspaceRevision,
    next_epoch: u64,
}

#[must_use]
pub(in crate::workspace::base) struct PrivateLiveRetirement {
    _old_tree: BindingTree,
    _old_hot: Option<PrivateHotBinding>,
    _expected_tree: BindingTree,
}

/// Sorted borrowed local bindings from the tree and its one hot override.
pub(in crate::workspace::base) struct PrivateLiveLocalIter<'a> {
    storage: &'a PrivateLiveStorage,
    previous: Option<&'a str>,
}

/// Sorted borrowed global spellings, including globals with absent values.
pub(in crate::workspace::base) struct PrivateLiveGlobalNames<'a> {
    storage: &'a PrivateLiveStorage,
    previous: Option<&'a str>,
}

impl<'a> Iterator for PrivateLiveLocalIter<'a> {
    type Item = (&'a str, &'a Value);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (name, binding) = self.storage.next_binding_after(self.previous)?;
            self.previous = Some(name);
            if let Some(BindingTreeValue::Local(entry)) = binding {
                return Some((name, entry.value()));
            }
        }
    }
}

impl<'a> Iterator for PrivateLiveGlobalNames<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (name, binding) = self.storage.next_binding_after(self.previous)?;
            self.previous = Some(name);
            if matches!(binding, Some(BindingTreeValue::Global(_))) {
                return Some(name);
            }
        }
    }
}

impl PreparedPrivateTransition {
    pub(in crate::workspace::base) fn next_revision(&self) -> WorkspaceRevision {
        self.next_revision
    }
}

impl PrivateLiveStorage {
    pub(in crate::workspace::base) fn empty(
        context: Option<&RuntimeContext>,
    ) -> Result<Self, HostAllocationError> {
        // Even an empty private owner must be issued by the current operation.
        HostAllocationReceipt::policy(context)?;
        Ok(Self {
            tree: BindingTree::default(),
            hot: None,
            revision: WorkspaceRevision::INITIAL,
            epoch: 0,
        })
    }

    /// Build one immutable root from an already retained logical generation.
    /// Local values share their physical generation and roots; exact global
    /// targets are retained even when their Session value is absent.
    pub(in crate::workspace::base) fn prepare_from_generation(
        source: &LogicalWorkspaceGeneration,
        context: Option<&RuntimeContext>,
    ) -> Result<Self, HostAllocationError> {
        let managed = matches!(
            HostAllocationReceipt::policy(context)?,
            HostAllocationPolicy::Managed
        );
        let local_count = source
            .iter()
            .filter(|(name, _)| source.global_target(name).is_none())
            .count();
        let global_count = source.global_bindings().count();
        let count = local_count
            .checked_add(global_count)
            .ok_or(HostAllocationError::Overflow)?;
        let mut retained = WorkspaceTableGeneration::retained_storage_bytes(count)?;
        for (name, _) in source.iter() {
            if source.global_target(name).is_none() {
                retained = retained
                    .checked_add(claimed_name_bytes(name)?)
                    .ok_or(HostAllocationError::Overflow)?;
                let lease = source
                    .read_lease(name)
                    .ok_or(HostAllocationError::StaleAuthority)?;
                if !lease.shared_generation().has_retained_claim() {
                    if !is_allocation_free_workspace_alias(lease.value()) && managed {
                        return Err(HostAllocationError::Resource(
                            crate::context::host_effect::HostEffectAllocationError::Unavailable,
                        ));
                    }
                    if managed {
                        let bytes = HostAllocationPlan::retained_exact_output_bytes(
                            u64::try_from(WorkspaceRetainedValue::requested_bytes())
                                .map_err(|_| HostAllocationError::Overflow)?,
                            1,
                        )?;
                        retained = retained
                            .checked_add(bytes)
                            .ok_or(HostAllocationError::Overflow)?;
                    }
                }
            }
        }
        for (name, target) in source.global_bindings() {
            let target_bytes = target.additional_retained_bytes()?;
            retained = retained
                .checked_add(claimed_name_bytes(name)?)
                .and_then(|bytes| bytes.checked_add(target_bytes))
                .ok_or(HostAllocationError::Overflow)?;
        }
        let tree = if count == 0 {
            HostAllocationReceipt::policy(context)?;
            BindingTree::default()
        } else {
            let receipt =
                HostAllocationReceipt::admit(context, HostAllocationPlan::new(0, retained)?)?;
            let permit = match (&receipt, context) {
                (Some(receipt), Some(context)) => Some(receipt.output_permit(context)?),
                _ => None,
            };
            let tree = Self::prepare_source_tree(source, count, permit.as_ref())?;
            if let (Some(receipt), Some(context)) = (&receipt, context) {
                receipt.checkpoint(context)?;
            }
            tree
        };
        Ok(Self {
            tree,
            hot: None,
            revision: source.base_revision(),
            epoch: 0,
        })
    }

    fn prepare_source_tree(
        source: &LogicalWorkspaceGeneration,
        count: usize,
        permit: Option<&HostOutputPermit>,
    ) -> Result<BindingTree, HostAllocationError> {
        let mut scratch = GenerationSortScratch::new(count, permit)?;
        for (name, _) in source.iter() {
            if source.global_target(name).is_some() {
                continue;
            }
            let lease = source
                .read_lease(name)
                .ok_or(HostAllocationError::StaleAuthority)?;
            let entry = if let Some(permit) = permit {
                if lease.shared_generation().has_retained_claim() {
                    RootedWorkspaceEntry::from_read_lease(WorkspaceSlotReadLease::from(lease))
                } else {
                    let retained = WorkspaceRetainedValue::prepare(lease.value(), permit)?.ok_or(
                        HostAllocationError::Resource(
                            crate::context::host_effect::HostEffectAllocationError::Unavailable,
                        ),
                    )?;
                    RootedWorkspaceEntry::from_retained(retained)
                }
            } else {
                RootedWorkspaceEntry::from_read_lease(WorkspaceSlotReadLease::from(lease))
            };
            let name = match permit {
                Some(permit) => WorkspaceName::prepare(name, permit)?,
                None => WorkspaceName::unbounded(name),
            };
            scratch.push(name, BindingTreeValue::Local(entry))?;
        }
        for (name, target) in source.global_bindings() {
            let name = match permit {
                Some(permit) => WorkspaceName::prepare(name, permit)?,
                None => WorkspaceName::unbounded(name),
            };
            let target = match permit {
                Some(permit) => target.prepare_retained(permit)?,
                None => target.clone(),
            };
            scratch.push(name, BindingTreeValue::Global(target))?;
        }
        if scratch.entries.len() != count {
            return Err(HostAllocationError::StaleAuthority);
        }
        Ok(scratch.finish(permit)?.0)
    }

    pub(in crate::workspace::base) fn revision(&self) -> WorkspaceRevision {
        self.revision
    }

    pub(in crate::workspace::base) fn epoch(&self) -> u64 {
        self.epoch
    }

    pub(in crate::workspace::base) fn len(&self) -> usize {
        self.tree.len() + usize::from(self.hot.as_ref().is_some_and(|hot| hot.binding.is_some()))
    }

    pub(in crate::workspace::base) fn is_empty(&self) -> bool {
        self.tree.is_empty() && self.hot.as_ref().is_none_or(|hot| hot.binding.is_none())
    }

    /// Transfer a sealed physical generation into the ordinary immutable
    /// Base snapshot representation. There is no payload/name clone or
    /// publication, and no detached namespace index.
    pub(in crate::workspace::base) fn sealed_managed_storage(
        &self,
    ) -> Result<ManagedWorkspaceStorage, HostAllocationError> {
        if self.hot.is_some() {
            return Err(HostAllocationError::StaleAuthority);
        }
        Ok(ManagedWorkspaceStorage {
            initialized: true,
            view_current: true,
            base: Some(WorkspaceTableGeneration(self.tree.clone())),
            committed: WorkspaceOverlay::default(),
            pending: WorkspaceOverlay::default(),
        })
    }

    pub(in crate::workspace::base) fn get(&self, name: &str) -> Option<&BindingTreeValue> {
        if let Some(hot) = &self.hot {
            if hot.name.as_str() == name {
                return hot.binding.as_ref();
            }
        }
        self.tree.get(name)
    }

    pub(in crate::workspace::base) fn local(&self, name: &str) -> Option<&RootedWorkspaceEntry> {
        match self.get(name) {
            Some(BindingTreeValue::Local(value)) => Some(value),
            Some(BindingTreeValue::Global(_)) | None => None,
        }
    }

    pub(in crate::workspace::base) fn global_target(
        &self,
        name: &str,
    ) -> Option<&GlobalBindingTarget> {
        match self.get(name) {
            Some(BindingTreeValue::Global(target)) => Some(target),
            Some(BindingTreeValue::Local(_)) | None => None,
        }
    }

    pub(in crate::workspace::base) fn iter_local(&self) -> PrivateLiveLocalIter<'_> {
        PrivateLiveLocalIter {
            storage: self,
            previous: None,
        }
    }

    pub(in crate::workspace::base) fn global_names(&self) -> PrivateLiveGlobalNames<'_> {
        PrivateLiveGlobalNames {
            storage: self,
            previous: None,
        }
    }

    pub(in crate::workspace::base) fn prepare_local(
        &self,
        context: Option<&RuntimeContext>,
        name: WorkspaceName,
        value: RootedWorkspaceEntry,
    ) -> Result<PreparedPrivateTransition, HostAllocationError> {
        self.prepare_change(context, name, Some(BindingTreeValue::Local(value)))
    }

    pub(in crate::workspace::base) fn prepare_global(
        &self,
        context: Option<&RuntimeContext>,
        name: WorkspaceName,
        target: GlobalBindingTarget,
    ) -> Result<PreparedPrivateTransition, HostAllocationError> {
        self.prepare_change(context, name, Some(BindingTreeValue::Global(target)))
    }

    pub(in crate::workspace::base) fn prepare_remove(
        &self,
        context: Option<&RuntimeContext>,
        name: WorkspaceName,
    ) -> Result<PreparedPrivateTransition, HostAllocationError> {
        self.prepare_change(context, name, None)
    }

    fn prepare_change(
        &self,
        context: Option<&RuntimeContext>,
        name: WorkspaceName,
        binding: Option<BindingTreeValue>,
    ) -> Result<PreparedPrivateTransition, HostAllocationError> {
        HostAllocationReceipt::policy(context)?;
        let next_revision = self.revision.next().ok_or(HostAllocationError::Overflow)?;
        let next_epoch = self
            .epoch
            .checked_add(1)
            .ok_or(HostAllocationError::Overflow)?;
        let (next_tree, next_hot) = self.prepare_binding_state(context, name, binding)?;
        Ok(PreparedPrivateTransition {
            expected_tree: self.tree.clone(),
            expected_revision: self.revision,
            expected_epoch: self.epoch,
            next_tree,
            next_hot,
            next_revision,
            next_epoch,
        })
    }

    fn prepare_binding_state(
        &self,
        context: Option<&RuntimeContext>,
        name: WorkspaceName,
        binding: Option<BindingTreeValue>,
    ) -> Result<(BindingTree, Option<PrivateHotBinding>), HostAllocationError> {
        let next_tree = if self.hot.as_ref().is_some_and(|hot| hot.name == name) {
            self.tree.clone()
        } else {
            let mut tree = self.tree.clone();
            if let Some(hot) = &self.hot {
                tree = match &hot.binding {
                    Some(value) => tree.prepare_set(context, hot.name.clone(), value.clone())?,
                    None => tree
                        .prepare_remove(context, hot.name.as_str())?
                        .unwrap_or(tree),
                };
            }
            tree.prepare_remove(context, name.as_str())?.unwrap_or(tree)
        };
        Ok((next_tree, Some(PrivateHotBinding { name, binding })))
    }

    /// Fold a completion's ordered actions in an unpublished local candidate.
    /// Intermediate path copies retire offstate; only the final tree/hot pair
    /// can be installed. One logical completion advances the public revision
    /// once, irrespective of its number of actions.
    pub(in crate::workspace::base) fn prepare_batch_with(
        &self,
        context: Option<&RuntimeContext>,
        len: usize,
        next: impl FnMut(
            usize,
        )
            -> Result<(WorkspaceName, Option<BindingTreeValue>), HostAllocationError>,
    ) -> Result<PreparedPrivateTransition, HostAllocationError> {
        self.prepare_batch_impl(context, len, next)
    }

    fn prepare_batch_impl(
        &self,
        context: Option<&RuntimeContext>,
        len: usize,
        mut next: impl FnMut(
            usize,
        )
            -> Result<(WorkspaceName, Option<BindingTreeValue>), HostAllocationError>,
    ) -> Result<PreparedPrivateTransition, HostAllocationError> {
        HostAllocationReceipt::policy(context)?;
        let mut candidate = self.clone();
        for index in 0..len {
            let (name, binding) = next(index)?;
            let (tree, hot) = candidate.prepare_binding_state(context, name, binding)?;
            candidate.tree = tree;
            candidate.hot = hot;
        }
        HostAllocationReceipt::policy(context)?;
        Ok(PreparedPrivateTransition {
            expected_tree: self.tree.clone(),
            expected_revision: self.revision,
            expected_epoch: self.epoch,
            next_tree: candidate.tree,
            next_hot: candidate.hot,
            next_revision: self.revision.next().ok_or(HostAllocationError::Overflow)?,
            next_epoch: self
                .epoch
                .checked_add(1)
                .ok_or(HostAllocationError::Overflow)?,
        })
    }

    pub(in crate::workspace::base) fn prepare_clear(
        &self,
        context: Option<&RuntimeContext>,
    ) -> Result<PreparedPrivateTransition, HostAllocationError> {
        HostAllocationReceipt::policy(context)?;
        Ok(PreparedPrivateTransition {
            expected_tree: self.tree.clone(),
            expected_revision: self.revision,
            expected_epoch: self.epoch,
            next_tree: BindingTree::default(),
            next_hot: None,
            next_revision: self.revision.next().ok_or(HostAllocationError::Overflow)?,
            next_epoch: self
                .epoch
                .checked_add(1)
                .ok_or(HostAllocationError::Overflow)?,
        })
    }

    pub(in crate::workspace::base) fn prepare_seal(
        &self,
        context: Option<&RuntimeContext>,
    ) -> Result<Option<PreparedPrivateTransition>, HostAllocationError> {
        HostAllocationReceipt::policy(context)?;
        let Some(hot) = &self.hot else {
            return Ok(None);
        };
        let next_tree = match &hot.binding {
            Some(value) => self
                .tree
                .prepare_set(context, hot.name.clone(), value.clone())?,
            None => self
                .tree
                .prepare_remove(context, hot.name.as_str())?
                .unwrap_or_else(|| self.tree.clone()),
        };
        Ok(Some(PreparedPrivateTransition {
            expected_tree: self.tree.clone(),
            expected_revision: self.revision,
            expected_epoch: self.epoch,
            next_tree,
            next_hot: None,
            next_revision: self.revision,
            next_epoch: self
                .epoch
                .checked_add(1)
                .ok_or(HostAllocationError::Overflow)?,
        }))
    }

    fn matches(&self, prepared: &PreparedPrivateTransition) -> bool {
        self.revision == prepared.expected_revision
            && self.epoch == prepared.expected_epoch
            && self.tree.same_root(&prepared.expected_tree)
    }

    fn install_prevalidated(
        &mut self,
        prepared: PreparedPrivateTransition,
    ) -> PrivateLiveRetirement {
        let old_tree = std::mem::replace(&mut self.tree, prepared.next_tree);
        let old_hot = std::mem::replace(&mut self.hot, prepared.next_hot);
        self.revision = prepared.next_revision;
        self.epoch = prepared.next_epoch;
        PrivateLiveRetirement {
            _old_tree: old_tree,
            _old_hot: old_hot,
            _expected_tree: prepared.expected_tree,
        }
    }

    /// The canonical owner adopts the projection's pruned tree after each
    /// private mutation, but never retains the projection's hot value. This
    /// releases an overwritten canonical root immediately after publication.
    pub(in crate::workspace::base) fn prepare_adopt_pruned(
        &self,
        context: Option<&RuntimeContext>,
        projection_change: &PreparedPrivateTransition,
    ) -> Result<PreparedPrivateTransition, HostAllocationError> {
        HostAllocationReceipt::policy(context)?;
        Ok(PreparedPrivateTransition {
            expected_tree: self.tree.clone(),
            expected_revision: self.revision,
            expected_epoch: self.epoch,
            next_tree: projection_change.next_tree.clone(),
            next_hot: None,
            next_revision: projection_change.next_revision,
            next_epoch: self
                .epoch
                .checked_add(1)
                .ok_or(HostAllocationError::Overflow)?,
        })
    }

    /// Atomic physical pair install for a canonical owner and its projection.
    /// Both destination-specific identities are validated before either
    /// state changes. The projection may have a hot suffix that the canonical
    /// owner never held; both prepared next roots must be the same sealed
    /// physical generation. The canonical next state has no hot value;
    /// projection keeps the one active hot value until its next transition.
    pub(in crate::workspace::base) fn install_pair_checked(
        canonical: &mut Self,
        projection: &mut Self,
        canonical_prepared: PreparedPrivateTransition,
        projection_prepared: PreparedPrivateTransition,
    ) -> Result<
        (PrivateLiveRetirement, PrivateLiveRetirement),
        (PreparedPrivateTransition, PreparedPrivateTransition),
    > {
        if !Self::pair_matches(
            canonical,
            projection,
            &canonical_prepared,
            &projection_prepared,
        ) {
            return Err((canonical_prepared, projection_prepared));
        }
        let canonical_retired = canonical.install_prevalidated(canonical_prepared);
        let projection_retired = projection.install_prevalidated(projection_prepared);
        Ok((canonical_retired, projection_retired))
    }

    /// Allocation-free read-only preflight for a short composite commit gate.
    /// Callers retain their outer issuer/revision/operation fences separately.
    pub(in crate::workspace::base) fn pair_matches(
        canonical: &Self,
        projection: &Self,
        canonical_prepared: &PreparedPrivateTransition,
        projection_prepared: &PreparedPrivateTransition,
    ) -> bool {
        !(canonical_prepared.next_hot.is_some()
            || canonical_prepared.next_revision != projection_prepared.next_revision
            || !canonical_prepared
                .next_tree
                .same_root(&projection_prepared.next_tree)
            || !canonical.matches(canonical_prepared)
            || !projection.matches(projection_prepared))
    }

    fn next_binding_after<'a>(
        &'a self,
        previous: Option<&str>,
    ) -> Option<(&'a str, Option<&'a BindingTreeValue>)> {
        let tree = self.tree.name_after(previous);
        let hot = self
            .hot
            .as_ref()
            .filter(|hot| previous.is_none_or(|prev| hot.name.as_str() > prev));
        match (tree, hot) {
            (None, None) => None,
            (Some((name, value)), None) => Some((name.as_str(), Some(value))),
            (None, Some(hot)) => Some((hot.name.as_str(), hot.binding.as_ref())),
            (Some((name, value)), Some(hot)) if name.as_str() < hot.name.as_str() => {
                Some((name.as_str(), Some(value)))
            }
            (Some(_), Some(hot)) => Some((hot.name.as_str(), hot.binding.as_ref())),
        }
    }

    pub(in crate::workspace::base) fn try_visit_sorted<E>(
        &self,
        mut visit: impl FnMut(&str, &BindingTreeValue) -> Result<(), E>,
    ) -> Result<(), E> {
        let mut previous = None;
        while let Some((name, binding)) = self.next_binding_after(previous) {
            previous = Some(name);
            if let Some(binding) = binding {
                visit(name, binding)?;
            }
        }
        Ok(())
    }

    pub(in crate::workspace::base) fn value(&self, name: &str) -> Option<&Value> {
        self.local(name).map(RootedWorkspaceEntry::value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn empty_root_aba_rejects_stale_pair_without_changing_either_owner() {
        let mut canonical = PrivateLiveStorage::empty(None).unwrap();
        let mut projection = canonical.clone();
        let stale_projection = projection.prepare_clear(None).unwrap();
        let stale_canonical = canonical
            .prepare_adopt_pruned(None, &stale_projection)
            .unwrap();

        let created = projection
            .prepare_local(
                None,
                WorkspaceName::unbounded("x"),
                RootedWorkspaceEntry::new(Value::Num(1.0)).unwrap(),
            )
            .unwrap();
        let owner_created = canonical.prepare_adopt_pruned(None, &created).unwrap();
        drop(
            PrivateLiveStorage::install_pair_checked(
                &mut canonical,
                &mut projection,
                owner_created,
                created,
            )
            .unwrap(),
        );
        {
            let mut locals = projection.iter_local();
            assert!(matches!(locals.next(), Some(("x", Value::Num(value))) if *value == 1.0));
            assert!(locals.next().is_none());
            assert!(projection.global_names().next().is_none());
        }
        let removed = projection
            .prepare_remove(None, WorkspaceName::unbounded("x"))
            .unwrap();
        let owner_removed = canonical.prepare_adopt_pruned(None, &removed).unwrap();
        drop(
            PrivateLiveStorage::install_pair_checked(
                &mut canonical,
                &mut projection,
                owner_removed,
                removed,
            )
            .unwrap(),
        );
        assert!(canonical.tree.is_empty() && projection.tree.is_empty());
        assert!(projection.value("x").is_none());
        assert!(projection.iter_local().next().is_none());
        assert!(projection.global_names().next().is_none());
        let revision = canonical.revision();
        let epoch = canonical.epoch();

        let stale = PrivateLiveStorage::install_pair_checked(
            &mut canonical,
            &mut projection,
            stale_canonical,
            stale_projection,
        );
        assert!(stale.is_err());
        drop(stale);
        assert_eq!(canonical.revision(), revision);
        assert_eq!(projection.revision(), revision);
        assert_eq!(canonical.epoch(), epoch);
        assert_eq!(projection.epoch(), epoch);
        assert!(projection.value("x").is_none());
    }
}
