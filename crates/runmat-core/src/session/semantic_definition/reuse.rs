use runmat_hir::FunctionId;
use runmat_vm::{Bytecode, FunctionRegistry};
use std::collections::{HashMap, HashSet};

/// Reuse published identities only after proving the complete incoming group
/// equivalent under one bijective mapping. Recursive/mutual references are
/// compared after all candidate identities are known, never one at a time.
pub(in crate::session) fn unchanged_unit_remap(
    bytecode: &Bytecode,
    incoming: &FunctionRegistry,
    published: &FunctionRegistry,
    private_names: &HashSet<String>,
) -> Option<HashMap<FunctionId, FunctionId>> {
    let mut remap = HashMap::new();
    let mut targets = HashSet::new();
    for (local, function) in &incoming.functions {
        // Private companions are execution-local and are deliberately not
        // published into the session's name authority.
        if private_names.contains(&function.display_name) {
            return None;
        }
        let existing_id = published.resolve_name(&function.display_name)?;
        let existing = published.get(existing_id)?;
        let source = function.source_id.or(bytecode.source_id)?;
        if existing.source_id != Some(source) || !targets.insert(existing_id) {
            return None;
        }
        remap.insert(*local, existing_id);
    }
    if let Some(layout) = &bytecode.layout {
        if layout
            .functions
            .keys()
            .any(|id| !remap.contains_key(id) && targets.contains(id))
        {
            return None;
        }
    }
    for (name, local) in &incoming.names {
        if published.resolve_name(name) != remap.get(local).copied() {
            return None;
        }
    }
    for (local, function) in &incoming.functions {
        let target = *remap.get(local)?;
        let mut normalized = function.clone();
        normalized.rebind_identity(
            target,
            runmat_types::ProgramFunctionId(u32::try_from(target.0).ok()?),
        );
        normalized.source_id = normalized.source_id.or(bytecode.source_id);
        for instruction in &mut normalized.instructions {
            super::super::compile::remap_semantic_function_instr(instruction, &remap);
        }
        if super::fingerprint(&normalized).ok()?
            != super::fingerprint(published.get(target)?).ok()?
        {
            return None;
        }
    }
    Some(remap)
}
