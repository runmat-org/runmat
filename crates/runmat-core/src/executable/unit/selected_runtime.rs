//! Bind a newly compiled script without rebinding its exact inherited bodies.

use std::collections::HashSet;

use runmat_hir::FunctionId;
use runmat_runtime::context::RuntimeContext;
use runmat_vm::{Bytecode, BytecodeSourceRef, FunctionRegistry};

use super::ExecutableUnit;

impl ExecutableUnit {
    pub(crate) fn runtime_bytecode_with_selected_prior(
        &self,
        runtime: &RuntimeContext,
        inherited: &FunctionRegistry,
        included_current: &HashSet<FunctionId>,
        partitioned_reused: &HashSet<FunctionId>,
        source_projection: &crate::session::SourceCompileProjection,
    ) -> Result<Bytecode, String> {
        let mut bytecode = self.bytecode().clone();
        let projected = bytecode.function_registry();
        let selected = FunctionRegistry::selected_execution_registry_from_prior(
            inherited,
            &projected,
            included_current,
            partitioned_reused,
            |source| source_projection.runtime_ref_for_program_ref(source),
        )?;
        for (function, body) in &selected.functions {
            if included_current.contains(function) {
                if body
                    .source_id
                    .is_none_or(|source| source.program().is_none())
                {
                    return Err("new selected definition lacks its compiler source ordinal".into());
                }
            } else {
                let source = body
                    .source_id
                    .and_then(BytecodeSourceRef::runtime)
                    .ok_or("inherited definition lacks its exact issued source")?;
                let owner = body
                    .selected_source_owner
                    .as_ref()
                    .ok_or("inherited definition lacks its selected source owner")?;
                if !owner.is_issued_in(runtime) || !owner.owns_source(source) {
                    return Err("inherited definition belongs to another source issuance".into());
                }
            }
        }
        // The compiler's bound mirror may contain speculative copies of
        // reused bodies. Replace it before source projection so every view
        // observes the same exact prior owner and selected product.
        bytecode.bound_functions = selected.functions.clone();
        bytecode.function_registry = selected;
        bytecode.project_source_ids(|source| match source {
            BytecodeSourceRef::Program(ordinal) => runtime
                .program_source_ref(ordinal)
                .map(BytecodeSourceRef::Runtime),
            BytecodeSourceRef::Runtime(reference) => reference
                .resolve_in(runtime)
                .map(|_| BytecodeSourceRef::Runtime(reference)),
        })?;
        let program = self
            .revision()
            .program_revision
            .identity_digest()
            .map_err(|error| error.to_string())?;
        let owner = self.selected_source_owner(runtime)?;
        bytecode
            .function_registry
            .attach_selected_program_source_owner(included_current.iter().copied(), program, &owner)
            .map_err(str::to_string)?;
        for function in included_current {
            let selected = bytecode
                .function_registry
                .get_current(*function)
                .ok_or("new selected definition vanished after source binding")?;
            let bound = bytecode
                .bound_functions
                .get_mut(function)
                .ok_or("new bound definition vanished after source binding")?;
            if selected.binding_origin != bound.binding_origin
                || !std::sync::Arc::ptr_eq(&selected.definition_owner, &bound.definition_owner)
                || selected.source_id != bound.source_id
            {
                return Err("new selected and bound definitions disagree".into());
            }
            bound.selected_source_owner = selected.selected_source_owner.clone();
        }
        Ok(bytecode)
    }
}
