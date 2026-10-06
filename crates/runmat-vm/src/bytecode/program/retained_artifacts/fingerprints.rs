use std::collections::{HashMap, HashSet};

use runmat_execution::{ExecutableSourceMapV2, FunctionBindingOrigin};
use runmat_hir::FunctionId;

use super::{function_remap, FunctionRegistry, RetainedCallableStageError};

#[path = "fingerprints/source_identity.rs"]
mod source_identity;
use source_identity::{project_definition_source, DefinitionSourceDomain};

/// Normalize session-local numbers to portable function ordinals before
/// comparing bodies from independently packaged executable units.
pub(super) fn normalized(
    registry: &FunctionRegistry,
) -> Result<HashMap<FunctionBindingOrigin, String>, RetainedCallableStageError> {
    normalized_in(registry, DefinitionSourceDomain::Runtime)
}

pub(super) fn normalized_admitted(
    registry: &FunctionRegistry,
    source_map: &ExecutableSourceMapV2,
) -> Result<HashMap<FunctionBindingOrigin, String>, RetainedCallableStageError> {
    normalized_in(registry, DefinitionSourceDomain::Admitted(source_map))
}

fn normalized_in(
    registry: &FunctionRegistry,
    source_domain: DefinitionSourceDomain<'_>,
) -> Result<HashMap<FunctionBindingOrigin, String>, RetainedCallableStageError> {
    let mut origins = HashSet::with_capacity(registry.functions.len());
    for body in registry.functions.values() {
        let origin = body
            .binding_origin
            .ok_or(RetainedCallableStageError::MissingOrigin {
                function: body.function,
            })?;
        if !origins.insert(origin) {
            return Err(RetainedCallableStageError::DuplicateOrigin);
        }
    }
    let mut fingerprints = HashMap::with_capacity(registry.functions.len());
    for body in registry.functions.values() {
        let origin = body
            .binding_origin
            .ok_or(RetainedCallableStageError::MissingOrigin {
                function: body.function,
            })?;
        let (portable_ids, referenced_origins) = direct_origin_ids(registry, body.function)?;
        let mut normalized = body.clone();
        function_remap::remap_closed_function(&mut normalized, &portable_ids);
        let source = project_definition_source(normalized.source_id, source_domain)?;
        // An ordinal is only a position in this admitted envelope. The exact
        // validated source record supplies immutable source identity below.
        normalized.source_id = None;
        let definition = super::super::semantic_definition_fingerprint(&normalized)?;
        let fingerprint = runmat_execution::Digest::sha256(serde_json::to_vec(&(
            definition,
            source,
            referenced_origins,
        ))?)
        .to_string();
        if fingerprints.insert(origin, fingerprint).is_some() {
            return Err(RetainedCallableStageError::DuplicateOrigin);
        }
    }
    Ok(fingerprints)
}

/// Project one immutable body and its direct semantic references into a dense
/// origin-ordered identity domain. Each referenced body's own fingerprint is
/// checked independently, so unrelated or merely transitive definitions must
/// not make this body's fingerprint quadratic or packaging-dependent.
fn direct_origin_ids(
    registry: &FunctionRegistry,
    root: FunctionId,
) -> Result<(HashMap<FunctionId, FunctionId>, Vec<FunctionBindingOrigin>), RetainedCallableStageError>
{
    let body = registry
        .functions
        .get(&root)
        .ok_or(RetainedCallableStageError::IncompleteDefinitionGraph)?;
    let mut referenced = HashSet::from([root]);
    for instruction in &body.instructions {
        function_remap::visit_instruction(instruction, |dependency| {
            referenced.insert(dependency);
        });
    }
    if referenced
        .iter()
        .any(|function| !registry.functions.contains_key(function))
    {
        return Err(RetainedCallableStageError::IncompleteDefinitionGraph);
    }
    let mut referenced = referenced.into_iter().collect::<Vec<_>>();
    referenced.sort_by_key(|function| {
        registry.functions[function]
            .binding_origin
            .map(|origin| (origin.program, origin.function))
    });
    let referenced_origins = referenced
        .iter()
        .map(|function| {
            registry.functions[function]
                .binding_origin
                .expect("all registry origins were validated")
        })
        .collect();
    let portable_ids = referenced
        .into_iter()
        .enumerate()
        .map(|(portable, function)| (function, FunctionId(portable)))
        .collect();
    Ok((portable_ids, referenced_origins))
}
