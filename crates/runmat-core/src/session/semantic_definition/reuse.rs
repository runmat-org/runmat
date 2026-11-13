use runmat_hir::FunctionId;
use runmat_vm::{Bytecode, FunctionRegistry};
use std::collections::{HashMap, HashSet};

/// A decorated product is a closed compilation: all its fresh-ID references,
/// including external and dotted encodings, belong to this current group.
/// Reuse published identities only after proving the complete incoming group
/// equivalent under one bijective mapping. Recursive/mutual references are
/// compared after all candidate identities are known, never one at a time.
pub(in crate::session) fn unchanged_closed_product_remap(
    bytecode: &Bytecode,
    incoming: &FunctionRegistry,
    published: &FunctionRegistry,
    private_names: &HashSet<String>,
) -> Option<HashMap<FunctionId, FunctionId>> {
    let mut remap = HashMap::new();
    let mut targets = HashSet::new();
    let mut sources = HashSet::new();
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
        sources.insert(source);
    }
    // Source identity normally includes the complete source text, but compiler
    // options can still change which definitions/aliases are published. Reuse
    // must prove equality, not merely that the incoming group is a subset.
    if published.functions.iter().any(|(id, function)| {
        function
            .source_id
            .is_some_and(|source| sources.contains(&source))
            && !targets.contains(id)
    }) {
        return None;
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
    for (name, target) in &published.names {
        if targets.contains(target)
            && incoming.names.get(name).and_then(|local| remap.get(local)) != Some(target)
        {
            return None;
        }
    }
    for (local, function) in &incoming.functions {
        let target = *remap.get(local)?;
        let mut normalized = function.clone();
        let source = normalized.source_id.or(bytecode.source_id);
        runmat_vm::rebind_session_source(&mut normalized, source).ok()?;
        runmat_vm::remap_closed_session_function(&mut normalized, target, &remap).ok()?;
        let incoming_fingerprint = super::fingerprint(&normalized);
        let published_fingerprint = super::fingerprint(published.get(target)?);
        if incoming_fingerprint.ok()? != published_fingerprint.ok()? {
            return None;
        }
    }
    Some(remap)
}
