use crate::ops::arithmetic as arithmetic_ops;
use crate::interpreter::dispatch::logical_truth_from_value;
use crate::ops::comparison as comparison_ops;
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;

pub async fn dispatch_arithmetic(
    instr: &crate::bytecode::Instr,
    stack: &mut Vec<Value>,
) -> Result<bool, RuntimeError> {
    match instr {
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
