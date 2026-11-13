use std::collections::HashSet;
use std::sync::Arc;

use runmat_hir::FunctionId;
use runmat_vm::{FunctionBytecode, FunctionRegistry};

/// Replace this successful compilation's publishable definitions with the
/// exact bodies decorated by the canonical executable owner.
///
/// Compilation prepares the session registry before executable regions,
/// coverage sites, and parallel metadata are installed. Validate the complete
/// selected set first, then use the registry's owning insertion path so source
/// indexes, execution-stack caches, aliases, and retained definition owners
/// remain coherent. Current private companions are intentionally absent from
/// `publication` and therefore remain execution-only.
pub(in crate::session) fn finalize_current_definitions(
    publication: &mut FunctionRegistry,
    current: &HashSet<FunctionId>,
    executable: &FunctionRegistry,
) -> Result<(), String> {
    let mut selected = current.iter().copied().collect::<Vec<_>>();
    selected.sort_by_key(|function| function.0);

    let replacements = selected
        .into_iter()
        .filter_map(|function| {
            publication
                .get_current(function)
                .map(|published| (function, published))
        })
        .map(|(function, published)| {
            let exact = executable.get_current(function).ok_or_else(|| {
                format!("published function {function:?} is absent from the canonical executable")
            })?;
            validate_same_definition(function, published, exact, publication, executable)?;
            let mut replacement = exact.clone();
            // `runtime_bytecode` binds the portable source map into a fresh
            // invocation owner. Session publication must retain the durable
            // source authority prepared by the session source pool instead of
            // capturing that invocation-local lease.
            runmat_vm::rebind_session_source(&mut replacement, published.source_id)?;
            Ok(replacement)
        })
        .collect::<Result<Vec<_>, String>>()?;

    for replacement in replacements {
        publication.insert_replacing_name(replacement);
    }
    Ok(())
}

fn validate_same_definition(
    function: FunctionId,
    published: &FunctionBytecode,
    exact: &FunctionBytecode,
    publication: &FunctionRegistry,
    executable: &FunctionRegistry,
) -> Result<(), String> {
    if published.function != function || exact.function != function {
        return Err(format!(
            "published function {function:?} disagrees with its registry identity"
        ));
    }
    let published_origin = published.binding_origin.ok_or_else(|| {
        format!("published function {function:?} has no immutable binding origin")
    })?;
    if exact.binding_origin != Some(published_origin) {
        return Err(format!(
            "published function {function:?} disagrees with its executable binding origin"
        ));
    }
    if published.portable_function != exact.portable_function
        || published.display_name != exact.display_name
    {
        return Err(format!(
            "published function {function:?} disagrees with its executable identity facts"
        ));
    }
    if published.source_file != exact.source_file
        || published.source_role != exact.source_role
        || published.private_owner_scope != exact.private_owner_scope
    {
        return Err(format!(
            "published function {function:?} disagrees with its executable source facts"
        ));
    }
    let published_source = published
        .source_id
        .and_then(runmat_vm::BytecodeSourceRef::runtime)
        .ok_or_else(|| {
            format!("published function {function:?} has an invalid source authority domain")
        })?;
    let exact_source = exact
        .source_id
        .and_then(runmat_vm::BytecodeSourceRef::runtime)
        .ok_or_else(|| {
            format!("published function {function:?} has an invalid source authority domain")
        })?;
    if published_source.semantic_content_revision() != exact_source.semantic_content_revision() {
        return Err(format!(
            "published function {function:?} disagrees with its executable source content"
        ));
    }
    if !Arc::ptr_eq(&published.definition_owner, &exact.definition_owner) {
        return Err(format!(
            "published function {function:?} disagrees with its executable definition owner"
        ));
    }

    if aliases_for(publication, function) != aliases_for(executable, function) {
        return Err(format!(
            "published function {function:?} disagrees with its executable aliases"
        ));
    }

    let source_membership = |registry: &FunctionRegistry, body: &FunctionBytecode| {
        body.source_id.map_or(0, |source| {
            registry
                .source_functions
                .get(&source)
                .map_or(0, |functions| {
                    functions
                        .iter()
                        .filter(|candidate| **candidate == function)
                        .count()
                })
        })
    };
    if source_membership(publication, published) != 1 || source_membership(executable, exact) != 1 {
        return Err(format!(
            "published function {function:?} has inconsistent source membership"
        ));
    }
    Ok(())
}

fn aliases_for(registry: &FunctionRegistry, function: FunctionId) -> Vec<&str> {
    let mut aliases = registry
        .names
        .iter()
        .filter_map(|(name, target)| (*target == function).then_some(name.as_str()))
        .collect::<Vec<_>>();
    aliases.sort_unstable();
    aliases
}
