//! Inline prepared changes for the explicit unbounded workspace policy.

use super::*;

pub(in crate::workspace::base) enum UnboundedProjectionAction {
    Assignment {
        name: String,
        slot: Option<usize>,
        local: bool,
    },
    Namespace {
        name: String,
        action: WorkspaceNamespaceAction,
        target: Option<GlobalBindingTarget>,
    },
    Clear,
}

pub(in crate::workspace::base) struct PreparedUnboundedProjectionMutation {
    pub(in crate::workspace::base) action: UnboundedProjectionAction,
    pub(in crate::workspace::base) expected_revision: WorkspaceRevision,
    transition: PreparedUnboundedDeltaTransition,
    managed_storage: ManagedWorkspaceStorage,
}

impl PreparedUnboundedProjectionMutation {
    pub(in crate::workspace::base) fn before_generation(&self) -> u64 {
        self.transition.before_generation()
    }

    pub(in crate::workspace::base) fn after_generation(&self) -> u64 {
        self.transition.after_generation()
    }
}

impl BaseWorkspaceProjection {
    pub(in crate::workspace::base) fn reserve_unbounded_mutation_capacity(
        &mut self,
        prepared: &PreparedUnboundedProjectionMutation,
    ) -> Result<(), BaseWorkspaceError> {
        assert_eq!(self.revision, prepared.expected_revision);
        self.ordered_view_mut()?
            .contents
            .pending
            .reserve_unbounded_transition(&prepared.transition)
            .map_err(BaseWorkspaceError::HostAllocation)
    }

    pub(in crate::workspace::base) fn prepare_unbounded_assignment(
        &self,
        name: &str,
        value: Value,
        slot: Option<usize>,
        local: bool,
    ) -> Result<PreparedUnboundedProjectionMutation, BaseWorkspaceError> {
        let entry = RootedWorkspaceEntry::new(value)?;
        self.prepare_unbounded_assignment_entry_with_order(name, &entry, slot, local, true)
    }

    pub(in crate::workspace::base) fn prepare_unbounded_assignment_entry_with_order(
        &self,
        name: &str,
        entry: &RootedWorkspaceEntry,
        slot: Option<usize>,
        local: bool,
        coalesce_assignment: bool,
    ) -> Result<PreparedUnboundedProjectionMutation, BaseWorkspaceError> {
        let entry = entry.clone();
        let origin = if coalesce_assignment {
            WorkspaceMutationOrigin::Executor
        } else {
            WorkspaceMutationOrigin::Ordered
        };
        let mut managed_storage = self
            .ordered_view()?
            .contents
            .managed_storage
            .prepare_unbounded_capture(
                &self.ordered_view()?.contents.entries,
                &self.ordered_view()?.contents.global_bindings,
            )
            .map_err(BaseWorkspaceError::HostAllocation)?;
        let overlay_name = WorkspaceName::unbounded(name);
        let overlay_backed = local || !self.is_global(name);
        if local {
            managed_storage.stage_unbounded_namespace(
                overlay_name.clone(),
                WorkspaceNamespaceAction::UnbindGlobal,
                None,
            );
        }
        if overlay_backed {
            managed_storage.stage_unbounded_owned_assignment(overlay_name.clone(), entry.clone());
        }
        let assignment_name = if overlay_backed {
            WorkspaceEntryName::Prepared(overlay_name.clone())
        } else {
            WorkspaceEntryName::Owned(name.to_owned())
        };
        let assignment = match slot {
            Some(slot) => {
                WorkspaceChange::assign_executor_entry(assignment_name, entry.clone(), slot)
            }
            None => WorkspaceChange::assign_entry(assignment_name, entry.clone()),
        };
        let assignment = if overlay_backed {
            assignment.staged_in_managed_overlay()
        } else {
            assignment
        };
        let transition = if local {
            self.ordered_view()?
                .contents
                .pending
                .prepare_unbounded_transition(
                    WorkspaceChange::unbind_global_prepared(WorkspaceEntryName::Prepared(
                        overlay_name,
                    ))
                    .staged_in_managed_overlay(),
                    Some(assignment),
                    coalesce_assignment,
                    origin,
                )
        } else {
            self.ordered_view()?
                .contents
                .pending
                .prepare_unbounded_transition(assignment, None, coalesce_assignment, origin)
        }
        .map_err(BaseWorkspaceError::HostAllocation)?;
        Ok(PreparedUnboundedProjectionMutation {
            action: UnboundedProjectionAction::Assignment {
                name: name.to_owned(),
                slot,
                local,
            },
            expected_revision: self.revision,
            transition,
            managed_storage: managed_storage
                .prepare_live(None)
                .map_err(BaseWorkspaceError::HostAllocation)?,
        })
    }

    pub(in crate::workspace::base) fn prepare_unbounded_namespace(
        &self,
        name: &str,
        action: WorkspaceNamespaceAction,
        target: Option<&GlobalBindingTarget>,
        origin: WorkspaceMutationOrigin,
    ) -> Result<PreparedUnboundedProjectionMutation, BaseWorkspaceError> {
        let target = (action == WorkspaceNamespaceAction::BindGlobal).then(|| {
            target
                .cloned()
                .unwrap_or_else(|| GlobalBindingTarget::Named(WorkspaceName::unbounded(name)))
        });
        let mut managed_storage = self
            .ordered_view()?
            .contents
            .managed_storage
            .prepare_unbounded_capture(
                &self.ordered_view()?.contents.entries,
                &self.ordered_view()?.contents.global_bindings,
            )
            .map_err(BaseWorkspaceError::HostAllocation)?;
        let overlay_name = WorkspaceName::unbounded(name);
        let overlay_backed = action != WorkspaceNamespaceAction::Remove || !self.is_global(name);
        if overlay_backed {
            managed_storage.stage_unbounded_namespace(overlay_name.clone(), action, target.clone());
        }
        let change_name = if overlay_backed {
            WorkspaceEntryName::Prepared(overlay_name)
        } else {
            WorkspaceEntryName::Owned(name.to_owned())
        };
        let change = match action {
            WorkspaceNamespaceAction::Remove => WorkspaceChange::remove_prepared(change_name),
            WorkspaceNamespaceAction::BindGlobal => WorkspaceChange::bind_global_prepared(
                change_name,
                target.clone().expect("global target"),
            ),
            WorkspaceNamespaceAction::UnbindGlobal => {
                WorkspaceChange::unbind_global_prepared(change_name)
            }
        };
        let change = if overlay_backed {
            change.staged_in_managed_overlay()
        } else {
            change
        };
        let transition = self
            .ordered_view()?
            .contents
            .pending
            .prepare_unbounded_transition(change, None, false, origin)
            .map_err(BaseWorkspaceError::HostAllocation)?;
        Ok(PreparedUnboundedProjectionMutation {
            action: UnboundedProjectionAction::Namespace {
                name: name.to_owned(),
                action,
                target,
            },
            expected_revision: self.revision,
            transition,
            managed_storage: managed_storage
                .prepare_live(None)
                .map_err(BaseWorkspaceError::HostAllocation)?,
        })
    }

    pub(in crate::workspace::base) fn prepare_unbounded_clear(
        &self,
        origin: WorkspaceMutationOrigin,
    ) -> Result<PreparedUnboundedProjectionMutation, BaseWorkspaceError> {
        let mut managed_storage = self
            .ordered_view()?
            .contents
            .managed_storage
            .prepare_unbounded_capture(
                &self.ordered_view()?.contents.entries,
                &self.ordered_view()?.contents.global_bindings,
            )
            .map_err(BaseWorkspaceError::HostAllocation)?;
        managed_storage.stage_unbounded_clear();
        let transition = self
            .ordered_view()?
            .contents
            .pending
            .prepare_unbounded_transition(
                WorkspaceChange::clear().staged_in_managed_overlay(),
                None,
                false,
                origin,
            )
            .map_err(BaseWorkspaceError::HostAllocation)?;
        Ok(PreparedUnboundedProjectionMutation {
            action: UnboundedProjectionAction::Clear,
            expected_revision: self.revision,
            transition,
            managed_storage: managed_storage
                .prepare_live(None)
                .map_err(BaseWorkspaceError::HostAllocation)?,
        })
    }
}

impl OrderedProjectionMut<'_> {
    pub(in crate::workspace::base) fn install_unbounded_mutation(
        &mut self,
        prepared: PreparedUnboundedProjectionMutation,
    ) -> ProjectionRetirements {
        assert_eq!(*self.revision, prepared.expected_revision);
        assert_eq!(self.pending().generation(), prepared.before_generation());
        let ordered = &mut self.contents;
        let mut retired = ProjectionRetirements {
            entries: Some(std::mem::take(&mut ordered.entries)),
            globals: Some(std::mem::take(&mut ordered.global_bindings)),
            storage: Some(std::mem::replace(
                &mut ordered.managed_storage,
                prepared.managed_storage,
            )),
            ..ProjectionRetirements::default()
        };
        (retired.replaced_changes, retired.pending) = ordered
            .pending
            .install_unbounded_transition(prepared.transition);
        retired
    }
}
