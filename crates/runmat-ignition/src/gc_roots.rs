use runmat_builtins::Value;
use runmat_gc::{
    gc_register_root, gc_unregister_root, GlobalRoot, RootId, StackRoot, VariableArrayRoot,
};

/// RAII wrapper for GC root management during interpretation
pub struct InterpretContext {
    pub(crate) stack_root_id: Option<RootId>,
    pub(crate) vars_root_id: Option<RootId>,
    pub(crate) extra_root_ids: Vec<RootId>,
}

impl InterpretContext {
    pub fn new(stack: &Vec<Value>, vars: &Vec<Value>) -> Result<Self, String> {
        let stack_root = Box::new(unsafe {
            StackRoot::new(stack as *const Vec<Value>, "interpreter_stack".to_string())
        });
        let vars_root = Box::new(unsafe {
            VariableArrayRoot::new(vars as *const Vec<Value>, "interpreter_vars".to_string())
        });

        let stack_root_id = gc_register_root(stack_root)
            .map_err(|e| format!("Failed to register stack root: {e:?}"))?;
        let vars_root_id = gc_register_root(vars_root)
            .map_err(|e| format!("Failed to register vars root: {e:?}"))?;

        Ok(InterpretContext {
            stack_root_id: Some(stack_root_id),
            vars_root_id: Some(vars_root_id),
            extra_root_ids: Vec::new(),
        })
    }

    /// Register a snapshot of global values as a GC root for the duration of this context
    pub fn register_global_values(
        &mut self,
        values: Vec<Value>,
        description: &str,
    ) -> Result<(), String> {
        if values.is_empty() {
            return Ok(());
        }
        let root = Box::new(GlobalRoot::new(values, description.to_string()));
        let id =
            gc_register_root(root).map_err(|e| format!("Failed to register global root: {e:?}"))?;
        self.extra_root_ids.push(id);
        Ok(())
    }
}

impl Drop for InterpretContext {
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
