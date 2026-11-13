use runmat_value::Value;

use crate::BuiltinResult;

/// Invoke the user's derivative through the shared semantic callback service.
/// The solver does not cache function names or implement a second resolver.
pub(super) async fn derivative(
    handle: &Value,
    time: f64,
    state: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    let callback = crate::canonicalize_callback_handle_for_semantic_resolution(handle.clone())?;
    crate::call_feval_async_with_outputs(callback, &[Value::Num(time), state], 1).await
}
