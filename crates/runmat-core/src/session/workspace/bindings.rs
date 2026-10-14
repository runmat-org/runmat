//! Compiler slot metadata reconciled with the canonical workspace owner.

use super::*;

impl RunMatSession {
    /// Observe only successful canonical publications. The projection owns the
    /// values; this weak callback updates the session's existing ABI metadata
    /// without keeping the session alive through a retained executor service.
    pub(crate) fn workspace_publication_observer(
        &self,
        execution_slots: &HashMap<String, usize>,
    ) -> runmat_runtime::workspace::WorkspacePublicationObserver {
        use runmat_runtime::workspace::WorkspacePublicationOrigin;

        let bindings = Rc::downgrade(&self.workspace_bindings);
        let source = self.active_source_identity.clone();
        let def_path = source_binding_def_path(self.current_source_name());
        let session = self.abi_workspace_handle.0;
        let execution_slots = execution_slots.clone();
        runmat_runtime::workspace::WorkspacePublicationObserver::new(move |changes| {
            let Some(bindings) = bindings.upgrade() else {
                return Ok(());
            };
            let mut current = bindings.try_borrow_mut().map_err(|_| {
                build_runtime_error("workspace binding publication is reentrant")
                    .with_identifier("RunMat:WorkspaceBindingPublication")
                    .build()
            })?;
            for change in changes {
                if let WorkspacePublicationOrigin::ExecutorSlot { name, slot } = change {
                    if *slot == usize::MAX
                        || execution_slots
                            .get(name)
                            .is_none_or(|slot| *slot == usize::MAX)
                    {
                        return Err(build_runtime_error(
                            "source store has no representable compiled workspace slot",
                        )
                        .with_identifier("RunMat:WorkspaceBindingSlots")
                        .build());
                    }
                }
            }
            for change in changes {
                match change {
                    WorkspacePublicationOrigin::ExecutorSlot { name, .. } => {
                        let slot = execution_slots[name];
                        let key = current
                            .get(name)
                            .filter(|binding| {
                                !matches!(
                                    binding.key,
                                    crate::abi::WorkspaceBindingKey::Global { .. }
                                )
                            })
                            .map(|binding| binding.key.clone())
                            .unwrap_or_else(|| match source.clone() {
                                Some(source) => crate::abi::WorkspaceBindingKey::SourceBinding {
                                    source,
                                    def_path: def_path.clone(),
                                    binding: runmat_hir::BindingName(name.clone()),
                                },
                                None => crate::abi::WorkspaceBindingKey::Interactive {
                                    session,
                                    name: runmat_hir::BindingName(name.clone()),
                                },
                            });
                        current.insert(name.clone(), SessionWorkspaceBinding { key, slot });
                    }
                    WorkspacePublicationOrigin::DynamicEffect { name } => {
                        if let Some(binding) = current.get_mut(name) {
                            if matches!(binding.key, crate::abi::WorkspaceBindingKey::Global { .. })
                            {
                                binding.key = crate::abi::WorkspaceBindingKey::Interactive {
                                    session,
                                    name: runmat_hir::BindingName(name.clone()),
                                };
                            }
                        }
                    }
                    WorkspacePublicationOrigin::Retired { name } => {
                        current.remove(name);
                    }
                    WorkspacePublicationOrigin::Cleared => current.clear(),
                }
            }
            Ok(())
        })
    }

    pub(super) fn interactive_workspace_binding_key(
        &self,
        name: &str,
    ) -> crate::abi::WorkspaceBindingKey {
        crate::abi::WorkspaceBindingKey::Interactive {
            session: self.abi_workspace_handle.0,
            name: runmat_hir::BindingName(name.to_string()),
        }
    }

    /// Refresh compiler/VM slot metadata from a completed invocation without
    /// treating the register file as a second base-workspace authority.
    pub(crate) fn refresh_workspace_binding_slots(
        &mut self,
        completed: &HashMap<String, (usize, Value)>,
        global_names: &std::collections::HashSet<String>,
    ) {
        for (name, (slot, _)) in completed {
            let key = if global_names.contains(name) {
                self.global_workspace_binding_key(name)
            } else {
                self.workspace_bindings
                    .borrow()
                    .get(name)
                    .filter(|binding| {
                        !matches!(binding.key, crate::abi::WorkspaceBindingKey::Global { .. })
                    })
                    .map(|binding| binding.key.clone())
                    .unwrap_or_else(|| self.interactive_workspace_binding_key(name))
            };
            self.workspace_bindings
                .borrow_mut()
                .insert(name.clone(), SessionWorkspaceBinding { key, slot: *slot });
        }
    }

    /// Reconcile compiler-only slots with the canonical base and named-global
    /// owners. Neither a prior register file nor an active source label can
    /// publish values or invent the origin of an effect-imported name.
    pub(crate) fn sync_workspace_binding_metadata(&mut self) -> Result<(), RuntimeError> {
        let base = self.base_workspace_snapshot()?;
        let globals = self.session_global_binding_names()?;
        let mut names = base
            .iter()
            .filter(|(name, _)| !globals.contains(*name))
            .map(|(name, _)| name.to_string())
            .collect::<std::collections::BTreeSet<_>>();
        for name in &globals {
            if self.runtime_context.session_global_value(name)?.is_none() {
                return Err(build_runtime_error(
                    "declared global has no canonical workspace value",
                )
                .with_identifier("RunMat:WorkspaceBindingSlots")
                .build());
            }
            names.insert(name.clone());
        }
        let old = self.workspace_bindings.borrow();
        let mut next = HashMap::with_capacity(names.len());
        let mut used = HashSet::with_capacity(names.len());
        let largest_live = names
            .iter()
            .filter_map(|name| old.get(name).map(|binding| binding.slot))
            .max();
        if largest_live == Some(usize::MAX) {
            return Err(
                build_runtime_error("workspace binding slot space is exhausted")
                    .with_identifier("RunMat:WorkspaceBindingSlots")
                    .build(),
            );
        }
        let mut fresh = largest_live.map_or(Some(0), |slot| slot.checked_add(1));
        for name in &names {
            let prior = old.get(name);
            let slot = if let Some(prior) = prior.filter(|binding| used.insert(binding.slot)) {
                prior.slot
            } else {
                let slot = fresh.ok_or_else(|| {
                    build_runtime_error("workspace binding slot space is exhausted")
                        .with_identifier("RunMat:WorkspaceBindingSlots")
                        .build()
                })?;
                used.insert(slot);
                fresh = slot.checked_add(1);
                slot
            };
            let key = if globals.contains(name) {
                self.global_workspace_binding_key(name)
            } else {
                prior
                    .filter(|binding| {
                        !matches!(binding.key, crate::abi::WorkspaceBindingKey::Global { .. })
                    })
                    .map(|binding| binding.key.clone())
                    .unwrap_or_else(|| self.interactive_workspace_binding_key(name))
            };
            next.insert(name.clone(), SessionWorkspaceBinding { key, slot });
        }
        drop(old);
        *self.workspace_bindings.borrow_mut() = next;
        Ok(())
    }

    pub(crate) fn reconcile_workspace_global_binding_keys(
        &mut self,
        execution_slots: &HashMap<String, usize>,
        global_names: &std::collections::HashSet<String>,
    ) -> std::result::Result<(), RuntimeError> {
        for name in global_names {
            let Some(slot) = execution_slots.get(name).copied().or_else(|| {
                self.workspace_bindings
                    .borrow()
                    .get(name)
                    .map(|binding| binding.slot)
            }) else {
                continue;
            };
            let key = self.global_workspace_binding_key(name);
            self.workspace_bindings
                .borrow_mut()
                .insert(name.clone(), SessionWorkspaceBinding { key, slot });
        }

        let retired = self
            .workspace_bindings
            .borrow()
            .iter()
            .filter(|(name, binding)| {
                matches!(&binding.key, crate::abi::WorkspaceBindingKey::Global { .. })
                    && !global_names.contains(*name)
            })
            .map(|(name, binding)| (name.clone(), binding.slot))
            .collect::<Vec<_>>();
        for (name, slot) in retired {
            if self
                .base_workspace
                .lookup(&name)
                .map_err(runmat_runtime::workspace::base_workspace_error_to_runtime)?
                .is_some()
            {
                self.workspace_bindings.borrow_mut().remove(&name);
                let key = self.interactive_workspace_binding_key(&name);
                self.workspace_bindings
                    .borrow_mut()
                    .insert(name, SessionWorkspaceBinding { key, slot });
            } else {
                self.workspace_bindings.borrow_mut().remove(&name);
            }
        }
        Ok(())
    }
}
