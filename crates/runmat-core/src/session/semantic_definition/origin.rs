use runmat_execution::{ContractError, FunctionBindingOrigin, ProgramRevision};

use super::super::PreparedExecution;

impl PreparedExecution {
    /// Publish the immutable compile-time ordinal while leaving Core's
    /// session-local function ID free to change. Existing retained definitions
    /// keep their earlier origin when a later interactive unit is compiled.
    pub(crate) fn attach_function_origins(
        &mut self,
        revision: &ProgramRevision,
    ) -> Result<(), ContractError> {
        for id in &self.current_function_ids {
            let Some(function) = self.bytecode.function_registry.functions.get_mut(id) else {
                continue;
            };
            let origin = match function.binding_origin {
                Some(origin) => origin,
                None => FunctionBindingOrigin::from_revision(revision, function.portable_function)?,
            };
            function.binding_origin = Some(origin);
            if let Some(published) = self.function_registry_after_success.functions.get_mut(id) {
                published.binding_origin = Some(origin);
            }
        }
        self.bytecode.bound_functions = self.bytecode.function_registry.functions.clone();
        Ok(())
    }
}
