use super::*;
use runmat_value::Value;

#[cfg(test)]
#[path = "workspace_initialization_tests.rs"]
mod tests;

impl HostState {
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
            if let Some(value) =
                runmat_runtime::call::lexical::read_binding(&environment, binding.binding)?
            {
                if let Some(workspace) = &self.workspace {
                    workspace.observe_shared_local(slot, Some(value.clone()))?;
                }
                let refreshed = self.arena.insert(value);
                self.locals[slot] = refreshed;
                self.record_local_initialization(slot, refreshed);
                return Ok(refreshed);
            }
            if let Some(workspace) = &self.workspace {
                workspace.observe_shared_local(slot, None)?;
            }
            self.locals[slot] = NativeValueRef::NULL;
            return Ok(NativeValueRef::NULL);
        }
        let reference = self
            .locals
            .get(slot)
            .copied()
            .ok_or_else(|| NativeExecutorError::Host("native local is out of bounds".into()))?;
        let global_name = self.global_bindings.name_for_slot(slot);
        if global_name.is_none() && !self.persistent_bindings.contains_key(&slot) {
            return Ok(reference);
        }
        let authority = self.runtime.session_variable_authority();
        if let Some(name) = global_name {
            let revision = authority.global_revision()?;
            if self.global_seen_revision.get(&slot) == Some(&revision) {
                return Ok(reference);
            }
            // The session global, not an old SSA/local snapshot, owns this alias.
            let value = authority.global_value(&name)?;
            return self.replace_local_from_global(slot, value, revision);
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
        value: Option<Value>,
        revision: runmat_runtime::workspace::session::GlobalRevision,
    ) -> NativeExecutorResult<NativeValueRef> {
        let refreshed = self.replace_local_from_canonical(slot, value)?;
        self.global_seen_revision.insert(slot, revision);
        Ok(refreshed)
    }

    fn replace_local_from_canonical(
        &mut self,
        slot: usize,
        value: Option<Value>,
    ) -> NativeExecutorResult<NativeValueRef> {
        let value = value.unwrap_or_else(empty_workspace_value);
        let refreshed = self.arena.insert(value);
        if let Some(workspace) = &self.workspace {
            workspace.refresh_local(slot, self.arena.get(refreshed)?.clone())?;
        }
        self.locals[slot] = refreshed;
        self.record_local_initialization(slot, refreshed);
        Ok(refreshed)
    }

    pub(in crate::execute) fn declared_session_reference(
        &mut self,
        slot: usize,
    ) -> NativeExecutorResult<Option<NativeValueRef>> {
        (self.global_bindings.name_for_slot(slot).is_some()
            || self.persistent_bindings.contains_key(&slot))
        .then(|| self.local_reference(slot))
        .transpose()
    }

    pub(in crate::execute) fn assignment_root(
        &mut self,
        slot: usize,
        create_struct: bool,
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
        if reference.is_null() && create_struct {
            return Ok(Value::Struct(runmat_value::StructValue::new()));
        }
        self.arena.get(reference).cloned()
    }

    pub fn set_local(&mut self, slot: usize, value: NativeValueRef) -> NativeExecutorResult<()> {
        if self.locals.get(slot).is_none() {
            return Err(NativeExecutorError::Host(
                "native local is out of bounds".into(),
            ));
        }
        if !value.is_null() {
            if let Some(workspace) = &self.workspace {
                workspace.synchronize_local(slot, self.arena.get(value)?.clone())?;
            }
        }
        self.locals[slot] = value;
        if value.is_null() {
            self.global_seen_revision.remove(&slot);
            self.persistent_seen_revision.remove(&slot);
        }
        self.record_local_initialization(slot, value);
        self.synchronize_session_binding(slot, value)?;
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
                    runmat_runtime::call::lexical::remove_owned_binding(
                        environment,
                        binding.binding,
                    )?;
                } else {
                    runmat_runtime::call::lexical::remove_binding(environment, binding.binding)?;
                }
            } else {
                let value = self.arena.get(value)?.clone();
                if binding.kind == runmat_native_codegen::NativeLexicalBindingKind::Owned {
                    runmat_runtime::call::lexical::write_owned_binding(
                        environment,
                        binding.binding,
                        value,
                    )?;
                } else {
                    runmat_runtime::call::lexical::write_binding(
                        environment,
                        binding.binding,
                        value,
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
            && (self.global_bindings.name_for_slot(slot).is_some()
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
            .is_some_and(|workspace| workspace.needs_reconciliation())
        {
            self.reconcile_workspace()?;
        }
        Ok(())
    }

    pub fn reconcile_workspace(&mut self) -> NativeExecutorResult<()> {
        let Some(workspace) = &self.workspace else {
            return Ok(());
        };
        for (slot, value) in workspace.local_values() {
            let reference = value
                .map(|value| self.arena.insert(value))
                .unwrap_or(NativeValueRef::NULL);
            let local = self.locals.get_mut(slot).ok_or_else(|| {
                NativeExecutorError::Host("native workspace local is out of bounds".into())
            })?;
            *local = reference;
            self.global_seen_revision.remove(&slot);
            self.persistent_seen_revision.remove(&slot);
            self.record_local_initialization(slot, reference);
        }
        if let Some(workspace) = &self.workspace {
            workspace.mark_reconciled();
        }
        Ok(())
    }

    pub fn workspace_snapshot(&self) -> Option<super::super::workspace::NativeWorkspaceSnapshot> {
        self.workspace
            .as_ref()
            .and_then(|workspace| workspace.snapshot())
    }

    pub fn record_expression(&mut self, value: NativeValueRef) {
        self.last_expression = Some(value);
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
            workspace.declare_global_binding(slot, &name)?;
        } else {
            self.global_bindings.declare_named(slot, name.clone());
        }
        let value = self
            .runtime
            .session_variable_authority()
            .declare_global(&name)?;
        let reference = self.arena.insert(value);
        self.set_local(slot, reference)
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
        if self.global_bindings.name_for_slot(slot).is_some() {
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
        let reference = self.arena.insert(value);
        self.set_local(slot, reference)
    }

    fn synchronize_session_binding(
        &mut self,
        slot: usize,
        reference: NativeValueRef,
    ) -> NativeExecutorResult<()> {
        if reference.is_null() {
            return Ok(());
        }
        let value = self.arena.get(reference)?.clone();
        if let Some(name) = self.global_bindings.name_for_slot(slot) {
            let authority = self.runtime.session_variable_authority();
            authority.store_global(&name, value.clone())?;
            self.global_seen_revision
                .insert(slot, authority.global_revision()?);
        }
        if let Some(binding) = self.persistent_bindings.get(&slot) {
            let authority = self.runtime.session_variable_authority();
            authority.store_persistent_named(&binding.function, &binding.name, value)?;
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
