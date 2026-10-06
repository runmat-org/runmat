use runmat_execution::{ContractError, FunctionBindingOrigin, ProgramRevision};

use super::super::PreparedExecution;

impl PreparedExecution {
    /// Publish the immutable compile-time ordinal while leaving Core's
    /// session-local function ID free to change. Existing retained definitions
    /// keep their earlier origin when a later interactive unit is compiled.
    pub(crate) fn attach_function_origins(
        &mut self,
        revision: &ProgramRevision,
        source_map: &crate::ExecutableSourceMap,
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
            function.source_file = function.source_id.and_then(|source_id| {
                let ordinal = runmat_types::ProgramSourceId(u32::try_from(source_id.0).ok()?);
                source_map
                    .entries()
                    .binary_search_by_key(&ordinal, |entry| entry.source_id)
                    .ok()
                    .and_then(|index| source_map.entries().get(index))
                    .and_then(|entry| entry.full_path.clone())
            });
            if let Some(published) = self.function_registry_after_success.functions.get_mut(id) {
                published.binding_origin = Some(origin);
                published.source_file = function.source_file.clone();
            }
        }
        self.bytecode.bound_functions = self.bytecode.function_registry.functions.clone();
        Ok(())
    }
}
