//! Deferred syntax crosses only an actual issued native model protocol.
//! Public standard-substruct encoding cannot turn End into a string/value.
use super::*;

pub(super) async fn assign(
    runtime: Option<&RuntimeContext>,
    method: &ResolvedObjectMethod,
    base: Value,
    path: ObjectSubscriptPath,
    values: RootedValueSequence,
) -> Result<RootedValue, RuntimeError> {
    if runtime.is_some_and(RuntimeContext::is_cancelled) {
        return Err(crate::execution::ExecutionServiceError::Cancelled
            .into_runtime_error_in_context(runtime));
    }
    let Value::Object(object) = &base else {
        return Err(crate::object::indexing::unresolved_end());
    };
    let target =
        crate::builtins::math::optim::optimmodel::native_fields::Target::select(object, method)
            .ok_or_else(crate::object::indexing::unresolved_end)?;
    crate::context::host_allocation::require_unbounded_workspace(
        runtime, "native contextual-index protocol",
        "native class issuance and deferred invocation metadata have no complete finite workspace bound",
    )?;
    let issuance = match runtime {
        Some(runtime) => {
            crate::class_registry::issue_builtin_class_in_context(runtime, object.class_name())
        }
        None => crate::class_registry::issue_builtin_class_ordinary(object.class_name()),
    }
    .ok_or_else(|| {
        semantic_error(
            "StaleObjectProtocolResolution",
            "contextual-index protocol has no actual builtin class issuance",
        )
    })?;
    let [rhs] = values.as_slice() else {
        return Err(semantic_error(
            "CommaSeparatedListAssignmentArity",
            "native contextual-index assignment requires one value",
        ));
    };
    target
        .assign(runtime, &base, &path, rhs, &issuance, method)
        .await
}
