//! One exact chain-rule dispatch shared by dense and active-column schedules.
use super::*;
pub(super) async fn propagate(
    state: &mut State<'_>,
    constants: &Evaluated,
    handle: GcHandle,
    weights: Weights,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    let program = state.program;
    let policy = state.policy;
    let node = program
        .source
        .nodes
        .get(&handle)
        .expect("retained reverse node")
        .node();
    if let Some(residual) = program
        .objective
        .quadratic()
        .and_then(|evidence| evidence.atoms.residual(handle))
    {
        if !constants.retains_operands() {
            return Err(
                policy.invalid("atomic norm-square reverse requires actual candidate operands")
            );
        }
        let count = evaluation::element_count(state.shape(residual)?, policy)?;
        for (row, weight) in weights {
            if row != 0 {
                return Err(policy.invalid("atomic norm-square reverse has a nonscalar output row"));
            }
            for input in 0..count {
                if state.active_at(residual, input)? {
                    let value = constants.value(residual, input, policy)?.ok_or_else(|| {
                        policy.invalid("atomic norm-square residual is unassigned")
                    })?;
                    let slope = Rational::from_usize(2)
                        .mul(&value, policy, cooperation)
                        .await?;
                    state
                        .add(
                            residual,
                            input,
                            weight.mul(&slope, policy, cooperation).await?,
                            cooperation,
                        )
                        .await?;
                }
                cooperation.completed(1).await?;
            }
        }
        return Ok(false);
    }
    match &node.kind {
        Kind::Variable(Value::Object(object)) => {
            let Some(runmat_value::optimization_variable::OptimizationVariablePayload::Reference(
                selection,
            )) = object.optimization_variable_payload()
            else {
                return Err(policy.invalid("reverse variable lost its original reference"));
            };
            let variable = program.layout.variable(selection.definition(), policy)?;
            for (row, value) in weights {
                let source = selection.definition_index(row).ok_or_else(|| {
                    policy.invalid("reverse variable reference exceeds its definition")
                })?;
                state
                    .gradient
                    .add(variable.start + source, value, policy, cooperation)
                    .await?;
                cooperation.completed(1).await?;
            }
        }
        Kind::Labels { source } => {
            for (row, value) in weights {
                state.add(*source, row, value, cooperation).await?;
                cooperation.completed(1).await?;
            }
        }
        Kind::Unary { operator, operand } => {
            use runmat_types::OperatorKind as Op;
            let m = state.shape(*operand)?.first().copied().unwrap_or(1);
            let n = node.shape.first().copied().unwrap_or(1);
            for (row, value) in weights {
                let (row, value) = match operator {
                    Op::UnaryPlus => (row, value),
                    Op::UnaryMinus => (row, -value),
                    Op::Transpose | Op::ConjugateTranspose => (row / n + (row % n) * m, value),
                    _ => return Err(policy.invalid("unadmitted reverse unary operation")),
                };
                state.add(*operand, row, value, cooperation).await?;
                cooperation.completed(1).await?;
            }
        }
        Kind::Binary {
            operator,
            left,
            right,
        } => {
            if matches!(
                operator,
                runmat_types::OperatorKind::MatrixPower
                    | runmat_types::OperatorKind::ElementwisePower
            ) {
                let evidence = program
                    .facts
                    .get(&handle)
                    .and_then(|fact| fact.power.as_deref())
                    .ok_or_else(|| {
                        policy.invalid("reverse power lost original exponent evidence")
                    })?;
                power::propagate(
                    state,
                    constants,
                    power::Power {
                        source: *left,
                        operator: *operator,
                        exponents: evidence,
                        shape: &node.shape,
                    },
                    weights,
                    cooperation,
                )
                .await?;
                return Ok(false);
            }
            binary::propagate(
                state,
                constants,
                *left,
                *right,
                *operator,
                &node.shape,
                weights,
                cooperation,
            )
            .await?;
        }
        Kind::Index { source, indices } => {
            for (row, value) in weights {
                let source_row = *indices
                    .get(row)
                    .ok_or_else(|| policy.invalid("reverse index exceeds original selection"))?;
                state.add(*source, source_row, value, cooperation).await?;
                cooperation.completed(1).await?;
            }
        }
        Kind::Assign {
            source,
            replacement,
            indices,
            selection_shape,
        } => {
            edits::assign(
                state,
                *source,
                *replacement,
                indices,
                selection_shape,
                weights,
                cooperation,
            )
            .await?;
        }
        Kind::Concatenate {
            dimension,
            operands,
        } => {
            edits::concatenate(
                state,
                *dimension,
                operands,
                &node.shape,
                weights,
                cooperation,
            )
            .await?;
        }

        Kind::Builtin {
            function,
            operands,
            arguments,
        } => {
            let Some(operation) =
                super::super::operations::Operation::selected(function.0.as_str())
            else {
                return Err(policy.invalid("unadmitted reverse source function"));
            };
            if operation == super::super::operations::Operation::Dot {
                if operands.len() != 2 {
                    return Err(policy.invalid("reverse dot has invalid original arity"));
                }
                dot::propagate(
                    state,
                    constants,
                    operands[0],
                    operands[1],
                    arguments,
                    &node.shape,
                    weights,
                    cooperation,
                )
                .await?;
                return Ok(false);
            }
            if operands.len() != 1 {
                return Err(policy.invalid("reverse affine function has invalid original arity"));
            }
            if operation == super::super::operations::Operation::Reshape {
                for (row, value) in weights {
                    state.add(operands[0], row, value, cooperation).await?;
                    cooperation.completed(1).await?;
                }
            } else if operation == super::super::operations::Operation::Difference {
                difference::propagate(
                    state,
                    operands[0],
                    arguments,
                    &node.shape,
                    weights,
                    cooperation,
                )
                .await?;
            } else if matches!(
                operation,
                super::super::operations::Operation::Product
                    | super::super::operations::Operation::Cumprod
            ) {
                products::propagate(
                    state,
                    constants,
                    handle,
                    operation,
                    operands[0],
                    arguments,
                    &node.shape,
                    weights,
                    cooperation,
                )
                .await?;
            } else {
                linear::propagate(
                    state,
                    handle,
                    operation,
                    operands[0],
                    arguments,
                    &node.shape,
                    weights,
                    cooperation,
                )
                .await?;
            }
        }
        Kind::Unassigned | Kind::EmptyInitializer | Kind::Zero | Kind::Constant(_) => {}
        _ => return Err(policy.invalid("unadmitted reverse source operation")),
    }

    Ok(true)
}
