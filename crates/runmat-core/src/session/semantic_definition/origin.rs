use runmat_execution::{ContractError, FunctionBindingOrigin, ProgramRevision};
use runmat_hir::FunctionId;
use std::collections::{HashMap, HashSet};

use super::super::private_source_owner::CapturedPrivateSourceOwners;
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
        attach_current_function_origins(
            &self.current_function_ids,
            &HashMap::new(),
            &mut self.bytecode,
            &mut self.function_registry_after_success,
            revision,
            source_map,
        )
    }
}

/// The canonical current-body origin projection is shared by ordinary source
/// execution and the typed dynamic compiler handoff. Inherited bodies are not
/// assigned the new source's identity.
pub(super) fn attach_current_function_origins(
    current_function_ids: &HashSet<FunctionId>,
    remap: &HashMap<FunctionId, FunctionId>,
    bytecode: &mut runmat_vm::Bytecode,
    published: &mut runmat_vm::FunctionRegistry,
    revision: &ProgramRevision,
    source_map: &crate::ExecutableSourceMap,
) -> Result<(), ContractError> {
    let private_owners = CapturedPrivateSourceOwners::new(source_map);
    for id in current_function_ids {
        let Some(function) = bytecode.function_registry.functions.get_mut(id) else {
            continue;
        };
        let origin = match function.binding_origin {
            Some(origin) => origin,
            None => FunctionBindingOrigin::from_revision(revision, function.portable_function)?,
        };
        function.binding_origin = Some(origin);
        function.source_file = function.source_id.and_then(|source_id| {
            let ordinal = source_id.program()?;
            source_map
                .entries()
                .binary_search_by_key(&ordinal, |entry| entry.source_id)
                .ok()
                .and_then(|index| source_map.entries().get(index))
                .and_then(|entry| entry.full_path.clone())
        });
        function.private_source_owner = function.source_id.and_then(|id| private_owners.get(id));
        if let Some(body) = published.functions.get_mut(remap.get(id).unwrap_or(id)) {
            body.binding_origin = Some(origin);
            body.source_file = function.source_file.clone();
            body.private_source_owner = function.private_source_owner.clone();
        }
    }
    bytecode.bound_functions = bytecode.function_registry.functions.clone();
    Ok(())
}
