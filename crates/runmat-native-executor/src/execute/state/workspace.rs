use super::*;
use runmat_types::AssignmentCreationPolicy;
use runmat_value::Value;

#[cfg(test)]
#[path = "workspace_initialization_tests.rs"]
mod tests;

impl HostState {
    pub(in crate::execute) fn empty_cell_root() -> Value {
        Value::Cell(
            runmat_value::CellArray::new(Vec::new(), 0, 0)
                .expect("the canonical empty cell shape is valid"),
        )
    }

    /// Return a staged empty-cell carrier only for a genuinely unassigned
    /// binding. A declared global or persistent is present as an empty numeric
    /// matrix even before its first canonical payload store.
    pub(in crate::execute) fn sequence_root(
        &mut self,
        slot: usize,
        creation: runmat_mir::SequenceRootCreation,
    ) -> NativeExecutorResult<(Value, Option<NativeValueRef>)> {
        if self
            .missing_input_locals
            .iter()
            .any(|local| local.0 as usize == slot)
        {
            return Err(runmat_runtime::runtime_error::semantic_error(
                "NotEnoughInputs",
                "Not enough input arguments.",
            )
            .into());
        }
        let reference = self.local_reference(slot)?;
        let absent = if matches!(
            creation,
            runmat_mir::SequenceRootCreation::EmptyCell
                | runmat_mir::SequenceRootCreation::OrdinaryContainer(_)
        ) {
            self.sequence_binding_absent(slot, reference)?
        } else {
            false
        };
        if absent {
            let intent = match creation {
                runmat_mir::SequenceRootCreation::OrdinaryContainer(intent) => intent,
                _ => runmat_types::AssignmentChildCreation::CellArray,
            };
            runmat_runtime::sequence::assignment_container_in_context(Some(&self.runtime), intent)?
                .with_owned(|value| {
                    let seed = self.arena.insert(value.clone(), Some(&self.runtime))?;
                    Ok((value, Some(seed)))
                })
        } else {
            Ok((self.arena.get(reference)?.clone(), None))
        }
    }

    pub(in crate::execute) fn sequence_root_still_absent(
        &mut self,
        slot: usize,
    ) -> NativeExecutorResult<bool> {
        let reference = self.local_reference(slot)?;
        self.sequence_binding_absent(slot, reference)
    }

    fn sequence_binding_absent(
        &self,
        slot: usize,
        reference: NativeValueRef,
    ) -> NativeExecutorResult<bool> {
        if self.global_bindings.has_binding_for_slot(slot)
            || self.persistent_bindings.contains_key(&slot)
        {
            Ok(false)
        } else {
            Ok(reference.is_null())
        }
    }

    /// Initialization is a frame-state transition shared by ordinary writes,
    /// call/edge results, optimized writes, and workspace reconciliation.
    /// A NULL transfer is still unassigned and cannot satisfy an omitted input.
    fn record_local_initialization(&mut self, slot: usize, value: NativeValueRef) {
        if !value.is_null() {
            self.missing_input_locals
                .retain(|local| local.0 as usize != slot);
        }
    }

    pub(in crate::execute) fn local_reference(
        &mut self,
        slot: usize,
    ) -> NativeExecutorResult<NativeValueRef> {
        if let (Some(environment), Some(binding)) = (
            self.lexical_environment.clone(),
            self.function
                .lexical_bindings
                .iter()
                .find(|binding| binding.local.0 as usize == slot)
                .copied(),
        ) {
            if binding.kind == runmat_native_codegen::NativeLexicalBindingKind::Transit {
                return self.locals.get(slot).copied().ok_or_else(|| {
                    NativeExecutorError::Host("native transit local is out of bounds".into())
                });
            }
            if let Some(value) = runmat_runtime::call::lexical::read_binding_lease_with_context(
                &environment,
                binding.binding,
                &self.runtime,
            )? {
                let prior = self.locals.get(slot).copied().ok_or_else(|| {
                    NativeExecutorError::Host("native lexical local is out of bounds".into())
                })?;
                let refreshed =
                    self.arena
                        .reuse_or_insert_retained_lease(prior, value, Some(&self.runtime))?;
                self.locals[slot] = refreshed;
                self.record_local_initialization(slot, refreshed);
                return Ok(refreshed);
            }
            self.locals[slot] = NativeValueRef::NULL;
            return Ok(NativeValueRef::NULL);
        }
        let reference = self
            .locals
            .get(slot)
            .copied()
            .ok_or_else(|| NativeExecutorError::Host("native local is out of bounds".into()))?;
        let global_target = self.global_bindings.binding_handle_for_slot(slot);
        if global_target.is_none() && !self.persistent_bindings.contains_key(&slot) {
            return Ok(reference);
        }
        let authority = self.runtime.session_variable_authority();
        if let Some(target) = global_target {
            let revision = authority.global_revision()?;
            if self.global_seen_revision.get(&slot) == Some(&(target.clone(), revision)) {
                return Ok(reference);
            }
            // The session global, not an old SSA/local snapshot, owns this alias.
            let value = super::super::global_target::current_value(&authority, &target)?;
            return self.replace_local_from_global(slot, target, value, revision);
        }
        if let Some(binding) = self.persistent_bindings.get(&slot) {
            let revision = authority.persistent_revision()?;
            if self.persistent_seen_revision.get(&slot) == Some(&revision) {
                return Ok(reference);
            }
            let name = binding.name.clone();
            let owner_function = binding.function.clone();
            let previous_invalidation = binding.invalidation.clone();
            let invalidation = authority.persistent_invalidation_revision(&owner_function)?;
            let value = authority.persistent_named_value(&owner_function, &name)?;
            let absent = value.is_none();
            let refreshed = self.replace_local_from_canonical(slot, value)?;
            if let Some(binding) = self.persistent_bindings.get_mut(&slot) {
                binding.invalidated =
                    absent && (binding.invalidated || previous_invalidation != invalidation);
                binding.invalidation = invalidation;
            }
            self.persistent_seen_revision.insert(slot, revision);
            return Ok(refreshed);
        }
        Ok(reference)
    }

    fn replace_local_from_global(
        &mut self,
        slot: usize,
        target: runmat_runtime::workspace::GlobalBindingTarget,
        value: Option<Value>,
        revision: runmat_runtime::workspace::session::GlobalRevision,
    ) -> NativeExecutorResult<NativeValueRef> {
        let refreshed = self.replace_local_from_canonical(slot, value)?;
        self.global_seen_revision.insert(slot, (target, revision));
        Ok(refreshed)
    }

    fn replace_local_from_canonical(
        &mut self,
        slot: usize,
        value: Option<Value>,
    ) -> NativeExecutorResult<NativeValueRef> {
        let value = value.unwrap_or_else(empty_workspace_value);
        let refreshed = self.arena.insert(value, Some(&self.runtime))?;
        if let Some(workspace) = &self.workspace {
            workspace.refresh_local_with_context(
                &self.runtime,
                slot,
                self.arena.get(refreshed)?.clone(),
            )?;
        }
        self.locals[slot] = refreshed;
        self.record_local_initialization(slot, refreshed);
        Ok(refreshed)
    }

    pub(in crate::execute) fn declared_session_reference(
        &mut self,
        slot: usize,
    ) -> NativeExecutorResult<Option<NativeValueRef>> {
        (self.global_bindings.has_binding_for_slot(slot)
            || self.persistent_bindings.contains_key(&slot))
        .then(|| self.local_reference(slot))
        .transpose()
    }

    pub(in crate::execute) fn assignment_root(
        &mut self,
        slot: usize,
        creation: AssignmentCreationPolicy,
    ) -> NativeExecutorResult<Value> {
        if self
            .missing_input_locals
            .iter()
            .any(|local| local.0 as usize == slot)
        {
            return Err(runmat_runtime::runtime_error::semantic_error(
                "NotEnoughInputs",
                "Not enough input arguments.",
            )
            .into());
        }
        let reference = self.local_reference(slot)?;
        if reference.is_null() {
            match creation {
                AssignmentCreationPolicy::CreateStructFieldPath => {
                    return Ok(Value::Struct(runmat_value::StructValue::new()));
                }
                AssignmentCreationPolicy::CreateArrayByIndex => {
                    return runmat_value::Tensor::new(Vec::new(), vec![0, 0])
                        .map(Value::Tensor)
                        .map_err(|error| {
                            runmat_runtime::runtime_error::semantic_error(
                                "IndexAssignmentBaseInit",
                                format!("failed to initialize indexed-assignment base: {error}"),
                            )
                            .into()
                        });
                }
                AssignmentCreationPolicy::ExistingOnly
                | AssignmentCreationPolicy::CreateBinding
                | AssignmentCreationPolicy::Overloaded => {}
            }
        }
        self.arena.get(reference).cloned()
    }

    pub fn set_local(&mut self, slot: usize, value: NativeValueRef) -> NativeExecutorResult<()> {
        if self.locals.get(slot).is_none() {
            return Err(NativeExecutorError::Host(
                "native local is out of bounds".into(),
            ));
        }
        let prepared_global = if !value.is_null() {
            self.global_bindings
                .binding_handle_for_slot(slot)
                .map(|target| {
                    let prepared = super::super::global_target::prepare_store(
                        &self.runtime.session_variable_authority(),
                        &self.runtime,
                        &target,
                        self.arena.get(value)?.clone(),
                    )
                    .map_err(NativeExecutorError::from)?;
                    Ok::<_, NativeExecutorError>((target, prepared))
                })
                .transpose()?
        } else {
            None
        };
        if let Some((_, prepared)) = &prepared_global {
            prepared.checkpoint(&self.runtime)?;
        }
        if !value.is_null() {
            if let Some(workspace) = &self.workspace {
                workspace.synchronize_local_with_context(
                    &self.runtime,
                    slot,
                    self.arena.get(value)?.clone(),
                )?;
            }
        }
        self.locals[slot] = value;
        if value.is_null() {
            self.global_seen_revision.remove(&slot);
            self.persistent_seen_revision.remove(&slot);
        }
        self.record_local_initialization(slot, value);
        if let Some((target, prepared)) = prepared_global {
            prepared.commit();
            self.global_seen_revision.insert(
                slot,
                (
                    target,
                    self.runtime
                        .session_variable_authority()
                        .global_revision()?,
                ),
            );
        }
        self.synchronize_persistent_binding(slot, value)?;
        if let (Some(environment), Some(binding)) = (
            self.lexical_environment.as_ref(),
            self.function
                .lexical_bindings
                .iter()
                .find(|binding| binding.local.0 as usize == slot),
        ) {
            if binding.kind == runmat_native_codegen::NativeLexicalBindingKind::Transit {
                return Ok(());
            }
            if value.is_null() {
                if binding.kind == runmat_native_codegen::NativeLexicalBindingKind::Owned {
                    runmat_runtime::call::lexical::remove_owned_binding_with_context(
                        environment,
                        binding.binding,
                        &self.runtime,
                    )?;
                } else {
                    runmat_runtime::call::lexical::remove_binding_with_context(
                        environment,
                        binding.binding,
                        &self.runtime,
                    )?;
                }
            } else {
                let value = self.arena.get(value)?.clone();
                if binding.kind == runmat_native_codegen::NativeLexicalBindingKind::Owned {
                    runmat_runtime::call::lexical::write_owned_binding_with_context(
                        environment,
                        binding.binding,
                        value,
                        &self.runtime,
                    )?;
                } else {
                    runmat_runtime::call::lexical::write_binding_with_context(
                        environment,
                        binding.binding,
                        value,
                        &self.runtime,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// A block parameter transfers an already-evaluated SSA snapshot, not a
    /// new assignment to a declared global or persistent. A nested call may
    /// have changed canonical session storage since that snapshot was made.
    pub(in crate::execute) fn bind_edge_local(
        &mut self,
        slot: usize,
        value: NativeValueRef,
        inherited_snapshot: bool,
    ) -> NativeExecutorResult<NativeValueRef> {
        if inherited_snapshot
            && (self.global_bindings.has_binding_for_slot(slot)
                || self.persistent_bindings.contains_key(&slot)
                || self
                    .function
                    .lexical_bindings
                    .iter()
                    .any(|binding| binding.local.0 as usize == slot))
        {
            return self.local_reference(slot);
        }
        self.set_local(slot, value)?;
        Ok(value)
    }

    pub fn reconcile_external_workspace(&mut self) -> NativeExecutorResult<()> {
        if self
            .workspace
            .as_ref()
            .map(|workspace| workspace.needs_reconciliation_with_context(&self.runtime))
            .transpose()?
            .unwrap_or(false)
        {
            self.reconcile_workspace()?;
        }
        Ok(())
    }

    pub fn reconcile_workspace(&mut self) -> NativeExecutorResult<()> {
        let Some(workspace) = &self.workspace else {
            return Ok(());
        };
        // A callback may update the canonical owner while this frame is
        // suspended. Record the revision seen before refreshing slots; if a
        // reentrant effect advances it during the refresh, the next boundary
        // still sees a mismatch and reconciles again.
        let observed = workspace.current_revision_with_context(&self.runtime)?;
        for (slot, _) in workspace.local_slots() {
            let local = self.locals.get_mut(slot).ok_or_else(|| {
                NativeExecutorError::Host("native workspace local is out of bounds".into())
            })?;
            let reference = match workspace.lookup_local_lease(&self.runtime, slot)? {
                Some(value) => {
                    self.arena
                        .reuse_or_insert_retained_lease(*local, value, Some(&self.runtime))?
                }
                None => NativeValueRef::NULL,
            };
            *local = reference;
            self.global_seen_revision.remove(&slot);
            self.persistent_seen_revision.remove(&slot);
            if !reference.is_null() {
                self.missing_input_locals
                    .retain(|local| local.0 as usize != slot);
            }
        }
        if let Some(workspace) = &self.workspace {
            workspace.record_observed_revision(observed);
            workspace.mark_reconciled();
        }
        Ok(())
    }

    pub fn workspace_snapshot(
        &self,
    ) -> NativeExecutorResult<Option<super::super::workspace::NativeWorkspaceSnapshot>> {
        match &self.workspace {
            Some(workspace) => workspace.snapshot_with_context(&self.runtime),
            None => Ok(None),
        }
    }

    pub fn record_expression(&mut self, value: NativeValueRef) {
        self.last_expression = Some(value);
    }

    pub(in crate::execute) fn expression_reference(&self) -> Option<NativeValueRef> {
        self.last_expression
    }

    pub fn expression_result(&self) -> NativeExecutorResult<Option<runmat_value::Value>> {
        self.last_expression
            .map(|value| self.arena.get(value).cloned())
            .transpose()
    }

    pub fn declare_global(&mut self, slot: usize, name: String) -> NativeExecutorResult<()> {
        if self.persistent_bindings.contains_key(&slot) {
            return Err(NativeExecutorError::Host(
                "native local cannot be both global and persistent".into(),
            ));
        }
        if let Some(workspace) = self.runtime.service_ports().workspace().cloned() {
            workspace.declare_global_binding_with_context(&self.runtime, slot, &name)?;
        } else {
            let authority = self.runtime.session_variable_authority();
            let prepared_session =
                authority.prepare_declare_global_with_context(&self.runtime, &name)?;
            let prepared_alias = self
                .global_bindings
                .prepare_named(&self.runtime, slot, &name)
                .map_err(|cause| {
                    super::super::global_target::alias_admission_error(&self.runtime, cause)
                })?;
            prepared_session.checkpoint(&self.runtime)?;
            prepared_alias.checkpoint().map_err(|cause| {
                super::super::global_target::alias_admission_error(&self.runtime, cause)
            })?;
            prepared_session.commit();
            prepared_alias.commit();
        }
        let target = self
            .global_bindings
            .binding_handle_for_slot(slot)
            .ok_or_else(|| {
                NativeExecutorError::Host(
                    "native global declaration did not install its exact target".into(),
                )
            })?;
        let authority = self.runtime.session_variable_authority();
        let value = super::super::global_target::current_value(&authority, &target)?;
        let revision = authority.global_revision()?;
        self.replace_local_from_global(slot, target, value, revision)
            .map(drop)
    }

    pub fn declare_persistent(&mut self, slot: usize, name: String) -> NativeExecutorResult<()> {
        let owner_function = self.function.name.clone();
        self.declare_persistent_for_owner(slot, name, owner_function)
    }

    pub fn declare_persistent_for_owner(
        &mut self,
        slot: usize,
        name: String,
        owner_function: String,
    ) -> NativeExecutorResult<()> {
        if self.global_bindings.has_binding_for_slot(slot) {
            return Err(NativeExecutorError::Host(
                "native local cannot be both global and persistent".into(),
            ));
        }
        let authority = self.runtime.session_variable_authority();
        let invalidation = authority.persistent_invalidation_revision(&owner_function)?;
        self.persistent_bindings.insert(
            slot,
            runmat_runtime::workspace::session::PersistentBindingContinuation {
                name: name.clone(),
                function: owner_function.clone(),
                owner: authority.clone(),
                invalidation,
                invalidated: false,
            },
        );
        let value = authority
            .persistent_named_value(&owner_function, &name)?
            .unwrap_or_else(empty_workspace_value);
        let reference = self.arena.insert(value, Some(&self.runtime))?;
        self.set_local(slot, reference)
    }

    fn synchronize_persistent_binding(
        &mut self,
        slot: usize,
        reference: NativeValueRef,
    ) -> NativeExecutorResult<()> {
        if reference.is_null() {
            return Ok(());
        }
        let value = self.arena.get(reference)?.clone();
        if let Some(binding) = self.persistent_bindings.get(&slot) {
            let authority = self.runtime.session_variable_authority();
            authority.store_persistent_named_with_context(
                &self.runtime,
                &binding.function,
                &binding.name,
                value,
            )?;
            self.persistent_seen_revision
                .insert(slot, authority.persistent_revision()?);
            let invalidation = authority.persistent_invalidation_revision(&binding.function)?;
            if let Some(binding) = self.persistent_bindings.get_mut(&slot) {
                binding.invalidation = invalidation;
                binding.invalidated = false;
            }
        }
        Ok(())
    }
}
