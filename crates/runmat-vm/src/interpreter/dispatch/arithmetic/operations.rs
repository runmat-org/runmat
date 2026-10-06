use super::*;

pub(super) async fn add(stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    arithmetic_ops::add(
        stack,
        call_operator_method,
        call_right_operator_method_ordered,
        |a, b| async move {
            let (a_acc, b_acc) = accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
            runmat_runtime::call_builtin_async("plus", &[a_acc, b_acc]).await
        },
    )
    .await?;
    Ok(true)
}

pub(super) async fn sub(stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    arithmetic_ops::sub(
        stack,
        call_operator_method,
        call_right_operator_method_ordered,
        |a, b| async move {
            let (a_acc, b_acc) = accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
            runmat_runtime::call_builtin_async("minus", &[a_acc, b_acc]).await
        },
    )
    .await?;
    Ok(true)
}

pub(super) async fn mul(stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    arithmetic_ops::mul(
        stack,
        call_operator_method,
        call_right_operator_method_ordered,
        |a, b| async move {
            let (a_acc, b_acc) = accel_promote_binary(AutoBinaryOp::MatMul, &a, &b).await?;
            runmat_runtime::value_matmul(&a_acc, &b_acc).await
        },
    )
    .await?;
    Ok(true)
}

pub(super) async fn elem_mul(stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    arithmetic_ops::binary_method_ordered(
        stack,
        "times",
        call_operator_method,
        call_right_operator_method_ordered,
        |a, b| async move {
            let (a_acc, b_acc) = accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
            runmat_runtime::call_builtin_async("times", &[a_acc, b_acc]).await
        },
    )
    .await?;
    Ok(true)
}

pub(super) async fn elem_div(stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    arithmetic_ops::binary_method_ordered(
        stack,
        "rdivide",
        call_operator_method,
        call_right_operator_method_ordered,
        |a, b| async move {
            let (a_acc, b_acc) = accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
            runmat_runtime::call_builtin_async("rdivide", &[a_acc, b_acc]).await
        },
    )
    .await?;
    Ok(true)
}

pub(super) async fn elem_pow(stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    arithmetic_ops::power(
        stack,
        |obj, _method, arg| async move {
            runmat_runtime::call_builtin_async("power", &[obj, arg]).await
        },
        call_right_operator_method_ordered,
        |a, b| async move {
            let (a_acc, b_acc) = accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
            runmat_runtime::call_builtin_async("power", &[a_acc, b_acc]).await
        },
    )
    .await?;
    Ok(true)
}

pub(super) async fn elem_left_div(stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    arithmetic_ops::binary_method_ordered(
        stack,
        "ldivide",
        call_operator_method,
        call_right_operator_method_ordered,
        |a, b| async move {
            let (b_acc, a_acc) = accel_promote_binary(AutoBinaryOp::Elementwise, &b, &a).await?;
            runmat_runtime::call_builtin_async("rdivide", &[b_acc, a_acc]).await
        },
    )
    .await?;
    Ok(true)
}

pub(super) async fn right_div(stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    arithmetic_ops::binary_fallback(stack, |a, b| async move {
        arithmetic_ops::execute_right_division(
                    &a,
                    &b,
                    call_operator_method,
                    call_right_operator_method_ordered,
                    |lhs, rhs| async move {
                        let (lhs_acc, rhs_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &lhs, &rhs).await?;
                        runmat_runtime::call_builtin_async("rdivide", &[lhs_acc, rhs_acc]).await
                    },
                    |lhs, rhs| async move {
                        runmat_runtime::call_builtin_async("mrdivide", &[lhs, rhs]).await
                    },
                )
                .await
    })
    .await?;
    Ok(true)
}

pub(super) async fn left_div(stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    arithmetic_ops::binary_fallback(stack, |a, b| async move {
        arithmetic_ops::execute_left_division(
                    &a,
                    &b,
                    call_operator_method,
                    call_right_operator_method_ordered,
                    |lhs, rhs| async move {
                        let (rhs_acc, lhs_acc) =
                            accel_promote_binary(AutoBinaryOp::Elementwise, &rhs, &lhs).await?;
                        runmat_runtime::call_builtin_async("rdivide", &[rhs_acc, lhs_acc]).await
                    },
                    |lhs, rhs| async move {
                        runmat_runtime::call_builtin_async("mldivide", &[lhs, rhs]).await
                    },
                )
                .await
    })
    .await?;
    Ok(true)
}

pub(super) async fn neg(stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    arithmetic_ops::unary_arithmetic(stack, |value| async move {
        match &value {
            Value::Object(obj) => {
                let args = vec![Value::Object(obj.clone())];
                match runmat_runtime::call_builtin_async("uminus", &args).await {
                    Ok(v) => Ok(v),
                    Err(_) => {
                        runmat_runtime::call_builtin_async(
                            "times",
                            &[value.clone(), Value::Num(-1.0)],
                        )
                        .await
                    }
                }
            }
            _ => {
                runmat_runtime::call_builtin_async("times", &[value.clone(), Value::Num(-1.0)])
                    .await
            }
        }
    })
    .await?;
    Ok(true)
}

pub(super) async fn pow(stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    arithmetic_ops::power(
        stack,
        call_operator_method,
        call_right_operator_method_ordered,
        |a, b| async move {
            if matches!(
                (&a, &b),
                (Value::Symbolic(_), _)
                    | (_, Value::Symbolic(_))
                    | (Value::SymbolicArray(_), _)
                    | (_, Value::SymbolicArray(_))
            ) {
                let args = [a, b];
                return runmat_runtime::call_builtin_async("power", &args).await;
            }
            let (a_acc, b_acc) = accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
            runmat_runtime::power(&a_acc, &b_acc).map_err(RuntimeError::from)
        },
    )
    .await?;
    Ok(true)
}

pub(super) async fn u_plus(stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    arithmetic_ops::unary_arithmetic(stack, |value| async move {
        match &value {
            Value::Object(obj) => {
                let args = vec![Value::Object(obj.clone())];
                match runmat_runtime::call_builtin_async("uplus", &args).await {
                    Ok(v) => Ok(v),
                    Err(_) => Ok(value),
                }
            }
            _ => Ok(value),
        }
    })
    .await?;
    Ok(true)
}

pub(super) async fn transpose(stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    arithmetic_ops::unary(stack, |value| async move {
        let promoted = accel_promote_unary(AutoUnaryOp::Transpose, &value).await?;
        runmat_runtime::call_builtin_async("transpose", &[promoted]).await
    })
    .await?;
    Ok(true)
}

pub(super) async fn conjugate_transpose(stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    arithmetic_ops::unary(stack, |value| async move {
        let promoted = accel_promote_unary(AutoUnaryOp::Transpose, &value).await?;
        runmat_runtime::call_builtin_async("ctranspose", &[promoted]).await
    })
    .await?;
    Ok(true)
}

pub(super) async fn less_equal(stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    comparison_ops::relation_inverted(
        stack,
        comparison_ops::RelationInvertedSpec {
            name: "le",
            inverse_name: "gt",
            right_name: "ge",
            right_inverse_name: "lt",
            predicate: |aa, bb| aa <= bb,
        },
        call_operator_method,
        |name, a, b| async move { runmat_runtime::call_builtin_async(name, &[a, b]).await },
        |v, label| async move { logical_truth_from_value(&v, &label).await },
    )
    .await?;
    Ok(true)
}

pub(super) async fn less(stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    comparison_ops::relation(
        stack,
        "lt",
        "gt",
        |aa, bb| aa < bb,
        call_operator_method,
        |name, a, b| async move { runmat_runtime::call_builtin_async(name, &[a, b]).await },
    )
    .await?;
    Ok(true)
}

pub(super) async fn greater(stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    comparison_ops::relation(
        stack,
        "gt",
        "lt",
        |aa, bb| aa > bb,
        call_operator_method,
        |name, a, b| async move { runmat_runtime::call_builtin_async(name, &[a, b]).await },
    )
    .await?;
    Ok(true)
}

pub(super) async fn greater_equal(stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    comparison_ops::relation_inverted(
        stack,
        comparison_ops::RelationInvertedSpec {
            name: "ge",
            inverse_name: "lt",
            right_name: "le",
            right_inverse_name: "gt",
            predicate: |aa, bb| aa >= bb,
        },
        call_operator_method,
        |name, a, b| async move { runmat_runtime::call_builtin_async(name, &[a, b]).await },
        |v, label| async move { logical_truth_from_value(&v, &label).await },
    )
    .await?;
    Ok(true)
}

pub(super) async fn equal(stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    comparison_ops::equal(
        stack,
        call_operator_method,
        |name, a, b| async move { runmat_runtime::call_builtin_async(name, &[a, b]).await },
        |_v, _label| async move { Ok(false) },
    )
    .await?;
    Ok(true)
}

pub(super) async fn not_equal(stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    comparison_ops::not_equal(
        stack,
        call_operator_method,
        |name, a, b| async move { runmat_runtime::call_builtin_async(name, &[a, b]).await },
        |v, label| async move { logical_truth_from_value(&v, &label).await },
    )
    .await?;
    Ok(true)
}

pub(super) async fn logical_not(stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    let value = stack
        .pop()
        .ok_or_else(|| crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
    stack.push(runmat_runtime::call_builtin_async("not", &[value]).await?);
    Ok(true)
}

pub(super) async fn logical_and(stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    let rhs = stack
        .pop()
        .ok_or_else(|| crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
    let lhs = stack
        .pop()
        .ok_or_else(|| crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
    stack.push(runmat_runtime::call_builtin_async("and", &[lhs, rhs]).await?);
    Ok(true)
}

pub(super) async fn logical_or(stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    let rhs = stack
        .pop()
        .ok_or_else(|| crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
    let lhs = stack
        .pop()
        .ok_or_else(|| crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
    stack.push(runmat_runtime::call_builtin_async("or", &[lhs, rhs]).await?);
    Ok(true)
}
