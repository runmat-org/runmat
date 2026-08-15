use runmat_runtime::native::NativeValueRef;
use runmat_types::OperatorKind;
use runmat_value::Value;

use crate::NativeExecutorResult;

use super::state::HostState;

pub(super) fn evaluate(
    state: &mut HostState,
    operator: OperatorKind,
    arguments: Vec<Value>,
) -> NativeExecutorResult<NativeValueRef> {
    let name = operator
        .overload_name()
        .expect("short-circuit operators do not reach runtime evaluation");
    let future = async {
        if arguments.len() == 2
            && !matches!(arguments[0], Value::Object(_) | Value::HandleObject(_))
            && matches!(arguments[1], Value::Object(_) | Value::HandleObject(_))
        {
            return runmat_runtime::object::dispatch::call_rhs_object_operator_method_ordered(
                arguments[0].clone(),
                arguments[1].clone(),
                name,
            )
            .await;
        }
        runmat_runtime::call_builtin_async(name, &arguments).await
    };
    let value = super::sync::complete(&state.runtime, future, "operator evaluation")?;
    Ok(state.arena.insert(value))
}
