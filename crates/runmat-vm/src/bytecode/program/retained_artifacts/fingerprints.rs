use std::collections::{HashMap, HashSet};

use runmat_execution::FunctionBindingOrigin;
use runmat_hir::FunctionId;

use super::{function_remap, FunctionRegistry, RetainedCallableStageError};

/// Normalize session-local numbers to portable function ordinals before
/// comparing bodies from independently packaged executable units.
pub(super) fn normalized(
    registry: &FunctionRegistry,
) -> Result<HashMap<FunctionBindingOrigin, String>, RetainedCallableStageError> {
    let mut portable_ids = HashMap::with_capacity(registry.functions.len());
    let mut portable_functions = HashSet::with_capacity(registry.functions.len());
    for (source_id, body) in &registry.functions {
        let portable = usize::try_from(body.portable_function.0)
            .map_err(|_| RetainedCallableStageError::FunctionIdentityExhausted)?;
        if !portable_functions.insert(portable) {
            return Err(RetainedCallableStageError::DuplicateOrigin);
        }
        portable_ids.insert(*source_id, FunctionId(portable));
    }
    let mut fingerprints = HashMap::with_capacity(registry.functions.len());
    for body in registry.functions.values() {
        let origin = body
            .binding_origin
            .ok_or(RetainedCallableStageError::MissingOrigin {
                function: body.function,
            })?;
        let mut normalized = body.clone();
        function_remap::remap_closed_function(&mut normalized, &portable_ids);
        let fingerprint = super::super::semantic_definition_fingerprint(&normalized)?;
        if fingerprints.insert(origin, fingerprint).is_some() {
            return Err(RetainedCallableStageError::DuplicateOrigin);
        }
    }
    Ok(fingerprints)
}
