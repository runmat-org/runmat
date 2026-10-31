use crate::accel::auto_promote::{
    accel_promote_binary, accel_promote_unary, AutoBinaryOp, AutoUnaryOp,
};
use crate::interpreter::dispatch::logical_truth_from_value;
use crate::ops::arithmetic as arithmetic_ops;
use crate::ops::comparison as comparison_ops;
use runmat_runtime::execution::RootedValue;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

pub(super) type ArithmeticDispatchFuture<'a> =
    runmat_runtime::execution::RuntimeContinuationFuture<'a, ()>;

async fn call_operator_method_in_context(
    runtime: &runmat_runtime::context::RuntimeContext,
    obj: Value,
    method: &str,
    arg: Value,
) -> Result<runmat_runtime::object::dispatch::OperatorMethodDispatch, RuntimeError> {
    let _roots = crate::call::root_values([&obj, &arg])?;
    runmat_runtime::object::dispatch::try_call_object_operator_method_in_context(
        runtime, obj, method, arg,
    )
    .await
}

async fn call_right_operator_method_ordered_in_context(
    runtime: &runmat_runtime::context::RuntimeContext,
    lhs: Value,
    rhs: Value,
    method: &str,
) -> Result<runmat_runtime::object::dispatch::OperatorMethodDispatch, RuntimeError> {
    let _roots = crate::call::root_values([&lhs, &rhs])?;
    runmat_runtime::object::dispatch::try_call_rhs_object_operator_method_ordered_in_context(
        runtime, lhs, rhs, method,
    )
    .await
}

pub(super) fn suspend_arithmetic<'a>(
    instr: &crate::bytecode::Instr,
    stack: &'a mut crate::interpreter::stack::InterpreterStack,
    runtime: &'a runmat_runtime::context::RuntimeContext,
) -> Result<ArithmeticDispatchFuture<'a>, RuntimeError> {
    let call_operator_method =
        |obj, method, arg| call_operator_method_in_context(runtime, obj, method, arg);
    let call_right_operator_method_ordered =
        |lhs, rhs, method| call_right_operator_method_ordered_in_context(runtime, lhs, rhs, method);
    match instr {
        crate::bytecode::Instr::Add => {
            runmat_runtime::execution::admit_stack_mutation(runtime, async move {
                arithmetic_ops::add(
                    stack,
                    call_operator_method,
                    call_right_operator_method_ordered,
                    |a, b| async move {
                        let (a_acc, b_acc) =
                            accel_promote_binary(runtime, AutoBinaryOp::Elementwise, &a, &b)
                                .await?;
                        crate::call::call_builtin_async(runtime, "plus", &[a_acc, b_acc]).await
                    },
                )
                .await?;
                Ok(())
            })
        }
        crate::bytecode::Instr::Sub => {
            runmat_runtime::execution::admit_stack_mutation(runtime, async move {
                arithmetic_ops::sub(
                    stack,
                    call_operator_method,
                    call_right_operator_method_ordered,
                    |a, b| async move {
                        let (a_acc, b_acc) =
                            accel_promote_binary(runtime, AutoBinaryOp::Elementwise, &a, &b)
                                .await?;
                        crate::call::call_builtin_async(runtime, "minus", &[a_acc, b_acc]).await
                    },
                )
                .await?;
                Ok(())
            })
        }
        crate::bytecode::Instr::Mul => {
            runmat_runtime::execution::admit_stack_mutation(runtime, async move {
                arithmetic_ops::mul(
                    stack,
                    call_operator_method,
                    call_right_operator_method_ordered,
                    |a, b| async move {
                        let (a_acc, b_acc) =
                            accel_promote_binary(runtime, AutoBinaryOp::MatMul, &a, &b).await?;
                        RootedValue::new(
                            runmat_runtime::value_matmul_in_context(runtime, &a_acc, &b_acc)
                                .await?,
                        )
                        .map_err(|error| error.into_runtime_error())
                    },
                    |operands| {
                        runtime
                            .vm_fusion_host_route()
                            .map(|route| {
                                route.retire_numerical_operands(
                                    runtime,
                                    runmat_runtime::context::VmFusionHostInstruction::Intrinsic(
                                        runmat_types::OperatorKind::MatrixMultiply,
                                    ),
                                    operands,
                                )
                            })
                            .unwrap_or(Ok(()))
                    },
                )
                .await?;
                Ok(())
            })
        }
        crate::bytecode::Instr::ElemMul => {
            runmat_runtime::execution::admit_stack_mutation(runtime, async move {
                arithmetic_ops::binary_method_ordered(
                    stack,
                    "times",
                    call_operator_method,
                    call_right_operator_method_ordered,
                    |a, b| async move {
                        let (a_acc, b_acc) =
                            accel_promote_binary(runtime, AutoBinaryOp::Elementwise, &a, &b)
                                .await?;
                        crate::call::call_builtin_async(runtime, "times", &[a_acc, b_acc]).await
                    },
                )
                .await?;
                Ok(())
            })
        }
        crate::bytecode::Instr::ElemDiv => {
            runmat_runtime::execution::admit_stack_mutation(runtime, async move {
                arithmetic_ops::binary_method_ordered(
                    stack,
                    "rdivide",
                    call_operator_method,
                    call_right_operator_method_ordered,
                    |a, b| async move {
                        let (a_acc, b_acc) =
                            accel_promote_binary(runtime, AutoBinaryOp::Elementwise, &a, &b)
                                .await?;
                        crate::call::call_builtin_async(runtime, "rdivide", &[a_acc, b_acc]).await
                    },
                )
                .await?;
                Ok(())
            })
        }
        crate::bytecode::Instr::ElemPow => {
            runmat_runtime::execution::admit_stack_mutation(runtime, async move {
                arithmetic_ops::power(
                    stack,
                    call_operator_method,
                    call_right_operator_method_ordered,
                    |a, b| async move {
                        let (a_acc, b_acc) =
                            accel_promote_binary(runtime, AutoBinaryOp::Elementwise, &a, &b)
                                .await?;
                        crate::call::call_builtin_async(runtime, "power", &[a_acc, b_acc]).await
                    },
                )
                .await?;
                Ok(())
            })
        }
        crate::bytecode::Instr::ElemLeftDiv => {
            runmat_runtime::execution::admit_stack_mutation(runtime, async move {
                arithmetic_ops::binary_method_ordered(
                    stack,
                    "ldivide",
                    call_operator_method,
                    call_right_operator_method_ordered,
                    |a, b| async move {
                        let (a_acc, b_acc) =
                            accel_promote_binary(runtime, AutoBinaryOp::Elementwise, &a, &b)
                                .await?;
                        crate::call::call_builtin_async(
                            runtime,
                            runmat_builtins::LDIVIDE_CATALOG_ENTRY.identity.name,
                            &[a_acc, b_acc],
                        )
                        .await
                    },
                )
                .await?;
                Ok(())
            })
        }
        crate::bytecode::Instr::RightDiv => {
            runmat_runtime::execution::admit_stack_mutation(runtime, async move {
                arithmetic_ops::binary_fallback(stack, |a, b| async move {
                    arithmetic_ops::execute_right_division(
                        &a,
                        &b,
                        call_operator_method,
                        call_right_operator_method_ordered,
                        |lhs, rhs| async move {
                            let (lhs_acc, rhs_acc) = accel_promote_binary(
                                runtime,
                                AutoBinaryOp::Elementwise,
                                &lhs,
                                &rhs,
                            )
                            .await?;
                            crate::call::call_builtin_async(runtime, "rdivide", &[lhs_acc, rhs_acc])
                                .await
                        },
                        |lhs, rhs| async move {
                            crate::call::call_builtin_async(runtime, "mrdivide", &[lhs, rhs]).await
                        },
                    )
                    .await
                })
                .await?;
                Ok(())
            })
        }
        crate::bytecode::Instr::LeftDiv => {
            runmat_runtime::execution::admit_stack_mutation(runtime, async move {
                arithmetic_ops::binary_fallback(stack, |a, b| async move {
                    arithmetic_ops::execute_left_division(
                        &a,
                        &b,
                        call_operator_method,
                        call_right_operator_method_ordered,
                        |lhs, rhs| async move {
                            let (lhs_acc, rhs_acc) = accel_promote_binary(
                                runtime,
                                AutoBinaryOp::Elementwise,
                                &lhs,
                                &rhs,
                            )
                            .await?;
                            crate::call::call_builtin_async(
                                runtime,
                                runmat_builtins::LDIVIDE_CATALOG_ENTRY.identity.name,
                                &[lhs_acc, rhs_acc],
                            )
                            .await
                        },
                        |lhs, rhs| async move {
                            crate::call::call_builtin_async(runtime, "mldivide", &[lhs, rhs]).await
                        },
                    )
                    .await
                })
                .await?;
                Ok(())
            })
        }
        crate::bytecode::Instr::Neg => {
            runmat_runtime::execution::admit_stack_mutation(runtime, async move {
                if let Some(route) = runtime.vm_fusion_host_route().cloned() {
                    arithmetic_ops::unary_arithmetic_with_retirement(
                        stack,
                        true,
                        |value| async move {
                            runmat_runtime::unary_arithmetic::evaluate_in_context(
                                Some(runtime),
                                runmat_runtime::unary_arithmetic::UnaryArithmeticOperation::Minus,
                                value,
                            )
                            .await
                        },
                        |operands| {
                            route.retire_numerical_operands(
                                runtime,
                                runmat_runtime::context::VmFusionHostInstruction::Intrinsic(
                                    runmat_types::OperatorKind::UnaryMinus,
                                ),
                                operands,
                            )
                        },
                    )
                    .await?;
                } else {
                    arithmetic_ops::unary_arithmetic(stack, |value| async move {
                        runmat_runtime::unary_arithmetic::evaluate_in_context(
                            Some(runtime),
                            runmat_runtime::unary_arithmetic::UnaryArithmeticOperation::Minus,
                            value,
                        )
                        .await
                    })
                    .await?;
                }
                Ok(())
            })
        }
        crate::bytecode::Instr::Pow => {
            runmat_runtime::execution::admit_stack_mutation(runtime, async move {
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
                            return crate::call::call_builtin_async(runtime, "power", &args).await;
                        }
                        crate::call::call_builtin_async(runtime, "mpower", &[a, b]).await
                    },
                )
                .await?;
                Ok(())
            })
        }
        crate::bytecode::Instr::UPlus => {
            runmat_runtime::execution::admit_stack_mutation(runtime, async move {
                if let Some(route) = runtime.vm_fusion_host_route().cloned() {
                    arithmetic_ops::unary_arithmetic_with_retirement(
                        stack,
                        false,
                        |value| async move {
                            runmat_runtime::unary_arithmetic::evaluate_in_context(
                                Some(runtime),
                                runmat_runtime::unary_arithmetic::UnaryArithmeticOperation::Plus,
                                value,
                            )
                            .await
                        },
                        |operands| {
                            route.retire_numerical_operands(
                                runtime,
                                runmat_runtime::context::VmFusionHostInstruction::Intrinsic(
                                    runmat_types::OperatorKind::UnaryPlus,
                                ),
                                operands,
                            )
                        },
                    )
                    .await?;
                } else {
                    arithmetic_ops::unary_arithmetic(stack, |value| async move {
                        runmat_runtime::unary_arithmetic::evaluate_in_context(
                            Some(runtime),
                            runmat_runtime::unary_arithmetic::UnaryArithmeticOperation::Plus,
                            value,
                        )
                        .await
                    })
                    .await?;
                }
                Ok(())
            })
        }
        crate::bytecode::Instr::Transpose => {
            runmat_runtime::execution::admit_stack_mutation(runtime, async move {
                arithmetic_ops::unary(stack, |value| async move {
                    let promoted =
                        accel_promote_unary(runtime, AutoUnaryOp::Transpose, &value).await?;
                    crate::call::call_builtin_async(runtime, "transpose", &[promoted]).await
                })
                .await?;
                Ok(())
            })
        }
        crate::bytecode::Instr::ConjugateTranspose => {
            runmat_runtime::execution::admit_stack_mutation(runtime, async move {
                arithmetic_ops::unary(stack, |value| async move {
                    let promoted =
                        accel_promote_unary(runtime, AutoUnaryOp::Transpose, &value).await?;
                    crate::call::call_builtin_async(runtime, "ctranspose", &[promoted]).await
                })
                .await?;
                Ok(())
            })
        }
        crate::bytecode::Instr::LessEqual => {
            runmat_runtime::execution::admit_stack_mutation(runtime, async move {
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
                    |name, a, b| async move {
                        crate::call::call_builtin_async(runtime, name, &[a, b]).await
                    },
                    |v, label| async move { logical_truth_from_value(&v, &label).await },
                )
                .await?;
                Ok(())
            })
        }
        crate::bytecode::Instr::Less => {
            runmat_runtime::execution::admit_stack_mutation(runtime, async move {
                comparison_ops::relation(
                    stack,
                    "lt",
                    "gt",
                    |aa, bb| aa < bb,
                    call_operator_method,
                    |name, a, b| async move {
                        crate::call::call_builtin_async(runtime, name, &[a, b]).await
                    },
                )
                .await?;
                Ok(())
            })
        }
        crate::bytecode::Instr::Greater => {
            runmat_runtime::execution::admit_stack_mutation(runtime, async move {
                comparison_ops::relation(
                    stack,
                    "gt",
                    "lt",
                    |aa, bb| aa > bb,
                    call_operator_method,
                    |name, a, b| async move {
                        crate::call::call_builtin_async(runtime, name, &[a, b]).await
                    },
                )
                .await?;
                Ok(())
            })
        }
        crate::bytecode::Instr::GreaterEqual => {
            runmat_runtime::execution::admit_stack_mutation(runtime, async move {
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
                    |name, a, b| async move {
                        crate::call::call_builtin_async(runtime, name, &[a, b]).await
                    },
                    |v, label| async move { logical_truth_from_value(&v, &label).await },
                )
                .await?;
                Ok(())
            })
        }
        crate::bytecode::Instr::Equal => {
            runmat_runtime::execution::admit_stack_mutation(runtime, async move {
                comparison_ops::equal(
                    stack,
                    call_operator_method,
                    |name, a, b| async move {
                        crate::call::call_builtin_async(runtime, name, &[a, b]).await
                    },
                    |_v, _label| async move { Ok(false) },
                )
                .await?;
                Ok(())
            })
        }
        crate::bytecode::Instr::NotEqual => {
            runmat_runtime::execution::admit_stack_mutation(runtime, async move {
                comparison_ops::not_equal(
                    stack,
                    call_operator_method,
                    |name, a, b| async move {
                        crate::call::call_builtin_async(runtime, name, &[a, b]).await
                    },
                    |v, label| async move { logical_truth_from_value(&v, &label).await },
                )
                .await?;
                Ok(())
            })
        }
        crate::bytecode::Instr::LogicalNot => {
            runmat_runtime::execution::admit_stack_mutation(runtime, async move {
                let value = stack.pop().ok_or_else(|| {
                    crate::interpreter::errors::mex("StackUnderflow", "stack underflow")
                })?;
                let _roots = crate::call::root_values([&value])?;
                crate::call::call_builtin_async(runtime, "not", &[value])
                    .await?
                    .with_owned(|result| stack.try_push(result))?;
                Ok(())
            })
        }
        crate::bytecode::Instr::LogicalAnd => {
            runmat_runtime::execution::admit_stack_mutation(runtime, async move {
                let rhs = stack.pop().ok_or_else(|| {
                    crate::interpreter::errors::mex("StackUnderflow", "stack underflow")
                })?;
                let lhs = stack.pop().ok_or_else(|| {
                    crate::interpreter::errors::mex("StackUnderflow", "stack underflow")
                })?;
                let _roots = crate::call::root_values([&lhs, &rhs])?;
                crate::call::call_builtin_async(runtime, "and", &[lhs, rhs])
                    .await?
                    .with_owned(|result| stack.try_push(result))?;
                Ok(())
            })
        }
        crate::bytecode::Instr::LogicalOr => {
            runmat_runtime::execution::admit_stack_mutation(runtime, async move {
                let rhs = stack.pop().ok_or_else(|| {
                    crate::interpreter::errors::mex("StackUnderflow", "stack underflow")
                })?;
                let lhs = stack.pop().ok_or_else(|| {
                    crate::interpreter::errors::mex("StackUnderflow", "stack underflow")
                })?;
                let _roots = crate::call::root_values([&lhs, &rhs])?;
                crate::call::call_builtin_async(runtime, "or", &[lhs, rhs])
                    .await?
                    .with_owned(|result| stack.try_push(result))?;
                Ok(())
            })
        }
        _ => unreachable!("arithmetic route requires an arithmetic opcode"),
    }
}
