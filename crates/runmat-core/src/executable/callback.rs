//! Project the canonical callback selector onto one executable's exact owners.
use runmat_runtime::call::descriptor::{CallableTarget, RepeatableCallable};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::execution::RootedValue;
use runmat_runtime::RuntimeError;
use runmat_value::BoundFunctionAuthority;

mod normalization;

/// Authenticate first, then normalize a stale portable cache through one new
/// admitted header and the same selector. The caller receives no unchecked
/// binding constructor or alternative backend authority.
pub(crate) fn select_owned(
    unit: &super::ExecutableUnit,
    runtime: &RuntimeContext,
    registry: &runmat_vm::FunctionRegistry,
    callback: RootedValue,
) -> Result<(RepeatableCallable, runmat_hir::FunctionId), RuntimeError> {
    let selected = crate::callable_selection::select(runtime, registry, callback.clone())?;
    let function = owned_function(unit, registry, selected.selected_target()).ok_or_else(|| {
        runmat_runtime::runtime_error::semantic_error(
            "RunMat:UndefinedExecutableProcedure",
            "retained callable is not owned by this executable unit",
        )
    })?;
    let Some(callback) = normalization::normalize(runtime, registry, callback, function)? else {
        return Ok((selected, function));
    };
    let normalized = crate::callable_selection::select(runtime, registry, callback)?;
    if owned_function(unit, registry, normalized.selected_target()) != Some(function) {
        return Err(runmat_runtime::runtime_error::semantic_error(
            "UnavailableBoundFunction",
            "normalized callable differs from its authenticated executable definition",
        ));
    }
    Ok((normalized, function))
}

/// Names select source callbacks only through the canonical resolver above.
/// An already bound callback must retain the exact current definition graph,
/// or identify a portable origin actually present in this unit. A same-name
/// definition from another publication does not pass this projection.
pub(crate) fn owned_function(
    unit: &super::ExecutableUnit,
    registry: &runmat_vm::FunctionRegistry,
    selected: &CallableTarget,
) -> Option<runmat_hir::FunctionId> {
    let binding = match selected {
        CallableTarget::Bound(binding) | CallableTarget::BoundLexical { binding, .. } => binding,
        _ => return None,
    };
    let function = match &binding.authority {
        BoundFunctionAuthority::Local { .. } => {
            let function = runmat_hir::FunctionId(binding.function);
            let expected = registry.exact_bound_binding(function)?;
            (expected == *binding).then_some(function)?
        }
        BoundFunctionAuthority::Portable { origin } => {
            let mut candidates = registry
                .functions
                .iter()
                .filter_map(|(id, body)| (body.binding_origin == Some(*origin)).then_some(*id));
            let function = candidates.next()?;
            if candidates.next().is_some() {
                return None;
            }
            // Require the selected invocation's live source owner and complete
            // dependency graph, not just matching serialized origin fields.
            registry.exact_bound_binding(function)?;
            function
        }
        BoundFunctionAuthority::SelectedNative { .. } => return None,
    };
    unit.owns_callable(function).then_some(function)
}
