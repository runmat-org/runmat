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

    pub(in crate::execute) fn assignment_root(
        &self,
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
        let reference = self.locals.get(slot).copied().ok_or_else(|| {
            NativeExecutorError::Host("assignment root local is out of bounds".into())
        })?;
        if reference.is_null() && create_struct {
            return Ok(Value::Struct(runmat_value::StructValue::new()));
        }
        self.arena.get(reference).cloned()
    }

    pub fn set_local(&mut self, slot: usize, value: NativeValueRef) -> NativeExecutorResult<()> {
        let local = self
            .locals
            .get_mut(slot)
            .ok_or_else(|| NativeExecutorError::Host("native local is out of bounds".into()))?;
        *local = value;
        if !value.is_null() {
            if let Some(workspace) = &self.workspace {
                workspace.synchronize_local(slot, self.arena.get(value)?.clone())?;
            }
        }
        self.record_local_initialization(slot, value);
        self.synchronize_session_binding(slot, value)
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
        self.global_bindings.insert(slot, name.clone());
        let value = runmat_runtime::workspace::session::global_value(&name)
            .unwrap_or_else(empty_workspace_value);
        let reference = self.arena.insert(value);
        self.set_local(slot, reference)
    }

    pub fn declare_persistent(&mut self, slot: usize, name: String) -> NativeExecutorResult<()> {
        if self.global_bindings.contains_key(&slot) {
            return Err(NativeExecutorError::Host(
                "native local cannot be both global and persistent".into(),
            ));
        }
        self.persistent_bindings.insert(slot, name.clone());
        let value =
            runmat_runtime::workspace::session::persistent_named_value(&self.function.name, &name)
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
        if let Some(name) = self.global_bindings.get(&slot) {
            runmat_runtime::workspace::session::store_global_named(name, value.clone());
        }
        if let Some(name) = self.persistent_bindings.get(&slot) {
            runmat_runtime::workspace::session::store_persistent_named(
                &self.function.name,
                name,
                value,
            );
        }
        Ok(())
    }
}
