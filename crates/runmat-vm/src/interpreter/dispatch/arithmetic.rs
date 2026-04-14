use crate::ops::arithmetic as arithmetic_ops;
use crate::accel::auto_promote::{accel_promote_binary, accel_promote_unary, AutoBinaryOp, AutoUnaryOp};
use crate::interpreter::dispatch::logical_truth_from_value;
use crate::ops::comparison as comparison_ops;
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;

pub async fn dispatch_arithmetic(
    instr: &crate::bytecode::Instr,
    stack: &mut Vec<Value>,
) -> Result<bool, RuntimeError> {
    match instr {
        crate::bytecode::Instr::Add => {
            arithmetic_ops::add(
                stack,
                |obj, method, arg| async move {
                    let args = vec![obj, Value::String(method.to_string()), arg];
                    runmat_runtime::call_builtin_async("call_method", &args).await
                },
                |a, b| async move {
                    let (a_acc, b_acc) =
                        accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
                    runmat_runtime::call_builtin_async("plus", &[a_acc, b_acc]).await
                },
            )
            .await?;
            Ok(true)
        }
        crate::bytecode::Instr::Sub => {
            arithmetic_ops::sub(
                stack,
                |obj, method, arg| async move {
                    let args = vec![obj, Value::String(method.to_string()), arg];
                    runmat_runtime::call_builtin_async("call_method", &args).await
                },
                |obj, lhs| async move {
                    let class_name = match &obj {
                        Value::Object(o) => o.class_name.clone(),
                        _ => String::new(),
                    };
                    let qualified = format!("{}.minus", class_name);
                    runmat_runtime::call_builtin_async(&qualified, &[lhs, obj]).await
                },
                |a, b| async move {
                    let (a_acc, b_acc) =
                        accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
                    runmat_runtime::call_builtin_async("minus", &[a_acc, b_acc]).await
                },
            )
            .await?;
            Ok(true)
        }
        crate::bytecode::Instr::Mul => {
            arithmetic_ops::mul(
                stack,
                |obj, method, arg| async move {
                    let args = vec![obj, Value::String(method.to_string()), arg];
                    runmat_runtime::call_builtin_async("call_method", &args).await
                },
                |a, b| async move {
                    let (a_acc, b_acc) = accel_promote_binary(AutoBinaryOp::MatMul, &a, &b).await?;
                    runmat_runtime::matrix::value_matmul(&a_acc, &b_acc).await
                },
            )
            .await?;
            Ok(true)
        }
        crate::bytecode::Instr::ElemMul => {
            arithmetic_ops::binary_method(
                stack,
                "times",
                |obj, method, arg| async move {
                    let args = vec![obj, Value::String(method.to_string()), arg];
                    runmat_runtime::call_builtin_async("call_method", &args).await
                },
                |a, b| async move {
                    let (a_acc, b_acc) =
                        accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
                    runmat_runtime::call_builtin_async("times", &[a_acc, b_acc]).await
                },
            )
            .await?;
            Ok(true)
        }
        crate::bytecode::Instr::ElemDiv => {
            arithmetic_ops::binary_method(
                stack,
                "rdivide",
                |obj, method, arg| async move {
                    let args = vec![obj, Value::String(method.to_string()), arg];
                    runmat_runtime::call_builtin_async("call_method", &args).await
                },
                |a, b| async move {
                    let (a_acc, b_acc) =
                        accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
                    runmat_runtime::call_builtin_async("rdivide", &[a_acc, b_acc]).await
                },
            )
            .await?;
            Ok(true)
        }
        crate::bytecode::Instr::ElemPow => {
            arithmetic_ops::power(
                stack,
                |obj, _method, arg| async move {
                    runmat_runtime::call_builtin_async("power", &[obj, arg]).await
                },
                |a, b| async move {
                    let (a_acc, b_acc) =
                        accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
                    runmat_runtime::call_builtin_async("power", &[a_acc, b_acc]).await
                },
            )
            .await?;
            Ok(true)
        }
        crate::bytecode::Instr::ElemLeftDiv => {
            arithmetic_ops::binary_method(
                stack,
                "ldivide",
                |obj, method, arg| async move {
                    let args = vec![obj, Value::String(method.to_string()), arg];
                    runmat_runtime::call_builtin_async("call_method", &args).await
                },
                |a, b| async move {
                    let (b_acc, a_acc) =
                        accel_promote_binary(AutoBinaryOp::Elementwise, &b, &a).await?;
                    runmat_runtime::call_builtin_async("rdivide", &[b_acc, a_acc]).await
                },
            )
            .await?;
            Ok(true)
        }
        crate::bytecode::Instr::RightDiv => {
            arithmetic_ops::binary_fallback(stack, |a, b| async move {
                arithmetic_ops::execute_right_division(
                    &a,
                    &b,
                    |obj, method, arg| async move {
                        let args = vec![obj, Value::String(method.to_string()), arg];
                        runmat_runtime::call_builtin_async("call_method", &args).await
                    },
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
        crate::bytecode::Instr::LeftDiv => {
            arithmetic_ops::binary_fallback(stack, |a, b| async move {
                arithmetic_ops::execute_left_division(
                    &a,
                    &b,
                    |obj, method, arg| async move {
                        let args = vec![obj, Value::String(method.to_string()), arg];
                        runmat_runtime::call_builtin_async("call_method", &args).await
                    },
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
        crate::bytecode::Instr::Neg => {
            arithmetic_ops::unary(stack, |value| async move {
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
                        runmat_runtime::call_builtin_async(
                            "times",
                            &[value.clone(), Value::Num(-1.0)],
                        )
                        .await
                    }
                }
            })
            .await?;
            Ok(true)
        }
        crate::bytecode::Instr::Pow => {
            arithmetic_ops::power(
                stack,
                |obj, method, arg| async move {
                    let args = vec![obj, Value::String(method.to_string()), arg];
                    runmat_runtime::call_builtin_async("call_method", &args).await
                },
                |a, b| async move {
                    let (a_acc, b_acc) =
                        accel_promote_binary(AutoBinaryOp::Elementwise, &a, &b).await?;
                    runmat_runtime::power(&a_acc, &b_acc).map_err(RuntimeError::from)
                },
            )
            .await?;
            Ok(true)
        }
        crate::bytecode::Instr::UPlus => {
            arithmetic_ops::unary(stack, |value| async move {
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
        crate::bytecode::Instr::Transpose => {
            arithmetic_ops::unary(stack, |value| async move {
                let promoted = accel_promote_unary(AutoUnaryOp::Transpose, &value).await?;
                runmat_runtime::call_builtin_async("transpose", &[promoted]).await
            })
            .await?;
            Ok(true)
        }
        crate::bytecode::Instr::ConjugateTranspose => {
            arithmetic_ops::unary(stack, |value| async move {
                let promoted = accel_promote_unary(AutoUnaryOp::Transpose, &value).await?;
                runmat_runtime::call_builtin_async("ctranspose", &[promoted]).await
            })
            .await?;
            Ok(true)
        }
        crate::bytecode::Instr::LessEqual => {
            comparison_ops::relation_inverted(
                stack,
                "le",
                "gt",
                "ge",
                "lt",
                |aa, bb| aa <= bb,
                |obj, method, arg| async move {
                    let args = vec![obj, Value::String(method.to_string()), arg];
                    runmat_runtime::call_builtin_async("call_method", &args).await
                },
                |name, a, b| async move { runmat_runtime::call_builtin_async(name, &[a, b]).await },
                |v, label| async move { logical_truth_from_value(&v, &label).await },
            )
            .await?;
            Ok(true)
        }
        crate::bytecode::Instr::Less => {
            comparison_ops::relation(
                stack,
                "lt",
                "gt",
                |aa, bb| aa < bb,
                |obj, method, arg| async move {
                    let args = vec![obj, Value::String(method.to_string()), arg];
                    runmat_runtime::call_builtin_async("call_method", &args).await
                },
                |name, a, b| async move { runmat_runtime::call_builtin_async(name, &[a, b]).await },
            )
            .await?;
            Ok(true)
        }
        crate::bytecode::Instr::Greater => {
            comparison_ops::relation(
                stack,
                "gt",
                "lt",
                |aa, bb| aa > bb,
                |obj, method, arg| async move {
                    let args = vec![obj, Value::String(method.to_string()), arg];
                    runmat_runtime::call_builtin_async("call_method", &args).await
                },
                |name, a, b| async move { runmat_runtime::call_builtin_async(name, &[a, b]).await },
            )
            .await?;
            Ok(true)
        }
        crate::bytecode::Instr::GreaterEqual => {
            comparison_ops::relation_inverted(
                stack,
                "ge",
                "lt",
                "le",
                "gt",
                |aa, bb| aa >= bb,
                |obj, method, arg| async move {
                    let args = vec![obj, Value::String(method.to_string()), arg];
                    runmat_runtime::call_builtin_async("call_method", &args).await
                },
                |name, a, b| async move { runmat_runtime::call_builtin_async(name, &[a, b]).await },
                |v, label| async move { logical_truth_from_value(&v, &label).await },
            )
            .await?;
            Ok(true)
        }
        crate::bytecode::Instr::Equal => {
            comparison_ops::equal(
                stack,
                |obj, method, arg| async move {
                    let args = vec![obj, Value::String(method.to_string()), arg];
                    runmat_runtime::call_builtin_async("call_method", &args).await
                },
                |name, a, b| async move { runmat_runtime::call_builtin_async(name, &[a, b]).await },
                |_v, _label| async move { Ok(false) },
            )
            .await?;
            Ok(true)
        }
        crate::bytecode::Instr::NotEqual => {
            comparison_ops::not_equal(
                stack,
                |obj, method, arg| async move {
                    let args = vec![obj, Value::String(method.to_string()), arg];
                    runmat_runtime::call_builtin_async("call_method", &args).await
                },
                |name, a, b| async move { runmat_runtime::call_builtin_async(name, &[a, b]).await },
                |v, label| async move { logical_truth_from_value(&v, &label).await },
            )
            .await?;
            Ok(true)
        }
        _ => Ok(false),
    }
}
