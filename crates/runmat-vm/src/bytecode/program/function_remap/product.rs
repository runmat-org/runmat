//! Transactional remap of a decorated session product after Core's group proof.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use runmat_hir::FunctionId;

use super::super::{
    semantic_definition_fingerprint, source_index, Bytecode, FunctionBytecode, FunctionRegistry,
};
use super::policy::{remap_closed_session_entrypoint, remap_closed_session_function};

#[path = "product/metadata.rs"]
mod metadata;
#[cfg(test)]
#[path = "product/tests.rs"]
mod tests;

/// Rebind only the decorated frame/region owners of a still-unpublished eval
/// candidate. The eval compiler separately remaps its entrypoint instructions
/// and local function bodies; both transforms use the same checked ID map.
///
/// The caller owns `bytecode` by value and discards it on failure, so no
/// partially remapped product can become executable or enter a registry.
pub fn remap_decorated_eval_metadata(
    bytecode: &mut Bytecode,
    remap: &HashMap<FunctionId, FunctionId>,
) -> Result<(), String> {
    if remap.values().copied().collect::<HashSet<_>>().len() != remap.len() {
        return Err("eval function remap is not injective".into());
    }
    if remap
        .keys()
        .any(|source| !bytecode.function_registry.functions.contains_key(source))
    {
        return Err("eval function remap references an absent local body".into());
    }
    metadata::remap(bytecode, remap)?;
    runmat_vm_layout_remap(bytecode, remap)
}

/// Rebind only fresh current definitions. Reused target bodies may already
/// occur in the merged registry; their definition-owner lease survives the
/// replacement. A complete Core group proof is still required before calling
/// this transform. This function checks each collision again and publishes no
/// partial mutation on error.
pub fn remap_decorated_session_definitions(
    bytecode: &mut Bytecode,
    published: &FunctionRegistry,
    remap: &HashMap<FunctionId, FunctionId>,
) -> Result<(), String> {
    if remap.is_empty() {
        return Ok(());
    }
    if remap.values().copied().collect::<HashSet<_>>().len() != remap.len() {
        return Err("decorated function remap is not injective".into());
    }
    if remap
        .keys()
        .any(|source| remap.values().any(|target| target == source))
    {
        return Err("decorated function remap has overlapping source and target domains".into());
    }
    #[cfg(feature = "native-accel")]
    if bytecode.function_private_layout.is_some() {
        return Err("decorated unit remap cannot consume an active function projection".into());
    }
    if bytecode.bound_functions.len() != bytecode.function_registry.functions.len()
        || bytecode
            .function_registry
            .functions
            .iter()
            .any(|(id, body)| {
                bytecode.bound_functions.get(id).is_none_or(|bound| {
                    !Arc::ptr_eq(&body.definition_owner, &bound.definition_owner)
                })
            })
    {
        return Err("decorated bound functions do not mirror the selected registry".into());
    }
    let mut replacements = Vec::with_capacity(remap.len());
    let mut sources = remap.keys().copied().collect::<Vec<_>>();
    sources.sort_by_key(|function| function.0);
    for source_id in &sources {
        let destination = remap[source_id];
        let source = bytecode
            .function_registry
            .get_current(*source_id)
            .ok_or_else(|| format!("decorated source function {:?} is absent", source_id))?;
        let mut replacement = source.clone();
        remap_closed_session_function(&mut replacement, destination, remap)?;
        let prior = published
            .get_current(destination)
            .ok_or_else(|| format!("reused decorated function {destination:?} is absent"))?;
        validate_collision(&replacement, prior)?;
        replacement.definition_owner = Arc::clone(&prior.definition_owner);
        if let Some(existing) = bytecode.function_registry.get_current(destination) {
            validate_collision(&replacement, existing)?;
        }
        replacements.push((destination, replacement));
    }
    let mut next = bytecode.clone();
    metadata::remap(&mut next, remap)?;
    runmat_vm_layout_remap(&mut next, remap)?;
    for source in &sources {
        next.function_registry.functions.remove(source);
    }
    for (destination, replacement) in replacements {
        next.function_registry
            .functions
            .insert(destination, replacement);
    }
    for target in next.function_registry.names.values_mut() {
        if let Some(mapped) = remap.get(target) {
            *target = *mapped;
        }
    }
    if next
        .function_registry
        .names
        .values()
        .any(|target| !next.function_registry.functions.contains_key(target))
    {
        return Err("decorated registry contains an unresolved alias".into());
    }
    let source_functions = source_index(&next.function_registry.functions);
    next.function_registry.source_functions = source_functions;
    next.bound_functions = next.function_registry.functions.clone();
    remap_closed_session_entrypoint(&mut next, remap)?;
    *bytecode = next;
    Ok(())
}

fn validate_collision(
    replacement: &FunctionBytecode,
    published: &FunctionBytecode,
) -> Result<(), String> {
    if replacement.binding_origin != published.binding_origin
        || replacement.source_id != published.source_id
        || semantic_definition_fingerprint(replacement).map_err(|error| error.to_string())?
            != semantic_definition_fingerprint(published).map_err(|error| error.to_string())?
    {
        return Err("decorated function collision lacks exact semantic identity".into());
    }
    Ok(())
}

fn runmat_vm_layout_remap(
    bytecode: &mut Bytecode,
    remap: &HashMap<FunctionId, FunctionId>,
) -> Result<(), String> {
    if let Some(layout) = &mut bytecode.layout {
        crate::layout::remap_layout_function_ids(layout, remap)
            .map_err(|error| format!("decorated VM layout remap: {error:?}"))?;
    }
    Ok(())
}
