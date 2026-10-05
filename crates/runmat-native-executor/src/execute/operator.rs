use runmat_mir::MirOperand;
use runmat_runtime::native::NativeValueRef;
use runmat_types::OperatorKind;
use runmat_value::Value;

use crate::NativeExecutorResult;

use super::state::HostState;

pub(super) fn evaluate(
    state: &mut HostState,
    operator: OperatorKind,
    operands: &[&MirOperand],
) -> NativeExecutorResult<NativeValueRef> {
    let embedded = state.enter_embedded_call();
    let result = evaluate_inner(state, operator, operands, embedded.as_ref()).and_then(|value| {
        let retained = state.arena.get(value)?.clone();
        state.cache_embedded_call(embedded.as_ref(), &[retained]);
        Ok(value)
    });
    let finished = state.finish_embedded_call(embedded.as_ref());
    match (result, finished) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(value), Ok(())) => Ok(value),
    }
}

fn evaluate_inner(
    state: &mut HostState,
    operator: OperatorKind,
    operands: &[&MirOperand],
    embedded: Option<&super::state::EmbeddedOperationIdentity>,
) -> NativeExecutorResult<NativeValueRef> {
    if let Some(values) = super::call_suspension::take_completed(state, embedded)? {
        let [value] = <[Value; 1]>::try_from(values).map_err(|_| {
            crate::NativeExecutorError::Host(
                "resumed native operator did not produce exactly one value".into(),
            )
        })?;
        return Ok(state.arena.insert(value));
    }
    let arguments = operands
        .iter()
        .map(|operand| super::operand::materialize_operand(state, operand))
        .collect::<NativeExecutorResult<Vec<_>>>()?;
    let name = operator
        .overload_name()
        .expect("short-circuit operators do not reach runtime evaluation");
    let retained_roots = arguments
        .iter()
        .cloned()
        .map(|value| state.arena.insert(value))
        .collect();
    let runtime = state.runtime.clone();
    let future = async move {
        let unary = match operator {
            OperatorKind::UnaryPlus => {
                Some(runmat_runtime::unary_arithmetic::UnaryArithmeticOperation::Plus)
            }
            OperatorKind::UnaryMinus => {
                Some(runmat_runtime::unary_arithmetic::UnaryArithmeticOperation::Minus)
            }
            _ => None,
        };
        if let Some(operation) = unary {
            let [argument] = arguments.as_slice() else {
                return Err(runmat_runtime::runtime_error::semantic_error(
                    "NativeUnaryArity",
                    "unary arithmetic requires one operand",
                ));
            };
            return runmat_runtime::unary_arithmetic::evaluate_in_context(
                Some(&runtime),
                operation,
                argument.clone(),
            )
            .await;
        }
        if arguments.len() == 2
            && !matches!(arguments[0], Value::Object(_) | Value::HandleObject(_))
            && matches!(arguments[1], Value::Object(_) | Value::HandleObject(_))
        {
            return runmat_runtime::object::dispatch::call_rhs_object_operator_method_ordered_in_context(
                &runtime,
                arguments[0].clone(),
                arguments[1].clone(),
                name.display_name(),
            )
            .await;
        }
        runmat_runtime::call_builtin_async_in_context(&runtime, name.display_name(), &arguments)
            .await
    };
    let scoped_runtime = state.runtime.clone();
    let values = super::call_suspension::begin_with_roots(
        state,
        embedded.cloned(),
        retained_roots,
        Box::pin(async move { Ok(vec![scoped_runtime.scope(future).await?]) }),
    )?;
    let [value] = <[Value; 1]>::try_from(values).map_err(|_| {
        crate::NativeExecutorError::Host("native operator did not produce exactly one value".into())
    })?;
    Ok(state.arena.insert(value))
}
