use runmat_gc::{
    gc_register_value_root, gc_unregister_root, GlobalRoot, RootId, StackRoot, VariableArrayRoot,
};
use runmat_value::Value;
use std::marker::PhantomData;
use std::ptr::NonNull;

mod sequence_roots;
mod interpreter_vars_root;
mod managed_roots;
pub(crate) use managed_roots::ManagedRoots;
#[cfg(test)]
mod frame_tests;

type RootLifetime<'roots> = fn(
    &'roots crate::interpreter::vars::InterpreterVars,
    &'roots Vec<Value>,
    &'roots Vec<Value>,
    &'roots crate::interpreter::dispatch::SequenceState,
);

/// RAII wrapper for GC root management during interpretation.
pub struct InterpretContext<'roots> {
    pub(crate) stack_root_id: Option<RootId>,
    pub(crate) vars_root_id: Option<RootId>,
    pub(crate) extra_root_ids: Vec<RootId>,
    _roots: PhantomData<RootLifetime<'roots>>,
}

impl<'roots> InterpretContext<'roots> {
    /// Register the complete frame before returning its root owner. Like the
    /// existing stack/vars constructor, this does not hold a shared Rust borrow
    /// while the interpreter mutates the registered owners between scans.
    pub(crate) fn new_frame(
        stack: &'roots Vec<Value>,
        vars: &'roots crate::interpreter::vars::InterpreterVars,
        locals: &'roots Vec<Value>,
        sequence: &'roots crate::interpreter::dispatch::SequenceState,
    ) -> Result<Self, String> {
        let mut context = Self::new(stack, vars)?;
        // SAFETY: the runner keeps all four owner objects at stable frame
        // addresses until this context drops. Vec backing storage may change;
        // VariableArrayRoot follows the Vec owner, not its element pointer.
        // Scans borrow owners synchronously and never overlap their mutation.
        let locals = Box::new(unsafe {
            VariableArrayRoot::new(NonNull::from(locals), "interpreter_locals".into())
        });
        context
            .extra_root_ids
            .push(gc_register_value_root(locals).map_err(|error| error.to_string())?);
        let sequence = Box::new(unsafe { sequence_roots::SequenceRoot::new(sequence) });
        context
            .extra_root_ids
            .push(gc_register_value_root(sequence).map_err(|error| error.to_string())?);
        Ok(context)
    }

    pub fn new(
        stack: &'roots Vec<Value>,
        vars: &'roots crate::interpreter::vars::InterpreterVars,
    ) -> Result<Self, String> {
        let stack_root = Box::new(unsafe {
            StackRoot::new(NonNull::from(stack), "interpreter_stack".to_string())
        });
        let vars_root = Box::new(unsafe { interpreter_vars_root::InterpreterVarsRoot::new(vars) });

        let stack_root_id = gc_register_value_root(stack_root)
            .map_err(|e| format!("Failed to register stack root: {e:?}"))?;
        let vars_root_id = gc_register_value_root(vars_root)
            .map_err(|e| format!("Failed to register vars root: {e:?}"))?;

        Ok(InterpretContext {
            stack_root_id: Some(stack_root_id),
            vars_root_id: Some(vars_root_id),
            extra_root_ids: Vec::new(),
            _roots: PhantomData,
        })
    }

    pub fn register_global_values(
        &mut self,
        values: Vec<Value>,
        description: &str,
    ) -> Result<(), String> {
        if values.is_empty() {
            return Ok(());
        }
        let root = Box::new(GlobalRoot::new(values, description.to_string()));
        let id = gc_register_value_root(root)
            .map_err(|e| format!("Failed to register global root: {e:?}"))?;
        self.extra_root_ids.push(id);
        Ok(())
    }
}

impl Drop for InterpretContext<'_> {
    fn drop(&mut self) {
        if let Some(id) = self.stack_root_id.take() {
            let _ = gc_unregister_root(id);
        }
        if let Some(id) = self.vars_root_id.take() {
            let _ = gc_unregister_root(id);
        }
        while let Some(id) = self.extra_root_ids.pop() {
            let _ = gc_unregister_root(id);
        }
    }
}
