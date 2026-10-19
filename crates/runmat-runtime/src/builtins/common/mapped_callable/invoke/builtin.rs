use crate::execution::RootedValue;
use runmat_value::Value;

use super::super::CallableCallError;

pub(super) async fn invoke_builtin_dispatch(
    context: Option<&crate::context::RuntimeContext>,
    name: &str,
    arguments: &[Value],
    requested_outputs: usize,
) -> Result<RootedValue, CallableCallError> {
    let result = match context {
        Some(context) => {
            crate::call_builtin_async_with_outputs_in_context(
                context,
                name,
                arguments,
                requested_outputs,
            )
            .await
        }
        None => crate::call_builtin_async_with_outputs(name, arguments, requested_outputs).await,
    };
    result.map_err(CallableCallError::Runtime)
}
