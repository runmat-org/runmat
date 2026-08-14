use runmat_mir::{MirConstant, MirOperand, MirRvalue};
use runmat_runtime::native::NativeValueRef;
use runmat_value::Value;

use crate::{JitError, JitResult};

use super::state::HostState;

pub(super) fn evaluate_rvalue(
    state: &mut HostState,
    value: &MirRvalue,
) -> JitResult<Vec<NativeValueRef>> {
    match value {
        MirRvalue::Use(operand) => evaluate_operand(state, operand).map(|value| vec![value]),
        MirRvalue::Unary(operator, operand) => {
            let argument = materialize_operand(state, operand)?;
            super::operator::evaluate(state, *operator, vec![argument]).map(|value| vec![value])
        }
        MirRvalue::Binary(left, operator, right) => {
            let left = materialize_operand(state, left)?;
            let right = materialize_operand(state, right)?;
            super::operator::evaluate(state, *operator, vec![left, right]).map(|value| vec![value])
        }
        MirRvalue::ShortCircuit {
            left,
            op,
            right_temps,
            right,
        } => {
            let left = materialize_operand(state, left)?;
            let left_truth = logical_truth(state, &left, "short-circuit left operand")?;
            let short_circuited = match op {
                runmat_mir::MirShortCircuitOp::And => !left_truth,
                runmat_mir::MirShortCircuitOp::Or => left_truth,
            };
            let result = if short_circuited {
                left_truth
            } else {
                for statement in right_temps {
                    execute_embedded_statement(state, &statement.kind)?;
                }
                let right = materialize_operand(state, right)?;
                logical_truth(state, &right, "short-circuit right operand")?
            };
            Ok(vec![state.arena.insert(Value::Bool(result))])
        }
        MirRvalue::Range { start, step, end } => {
            let mut arguments = vec![materialize_operand(state, start)?];
            if let Some(step) = step {
                arguments.push(materialize_operand(state, step)?);
            }
            arguments.push(materialize_operand(state, end)?);
            super::call::builtin(state, "colon", arguments, 1).map(|values| {
                values
                    .into_iter()
                    .map(|value| state.arena.insert(value))
                    .collect()
            })
        }
        MirRvalue::Call(call) => super::call::evaluate(state, call).map(|values| {
            values
                .into_iter()
                .map(|value| state.arena.insert(value))
                .collect()
        }),
        MirRvalue::Aggregate {
            kind,
            rows,
            cols,
            elements,
        } => super::aggregate::evaluate(state, kind, *rows, *cols, elements)
            .map(|value| vec![state.arena.insert(value)]),
        MirRvalue::StructLiteral { fields } => {
            super::aggregate::structure(state, fields).map(|value| vec![state.arena.insert(value)])
        }
        MirRvalue::ObjectLiteral { class_name, fields } => {
            super::aggregate::object(state, class_name, fields)
                .map(|value| vec![state.arena.insert(value)])
        }
        other => Err(JitError::UnsupportedSite(format!(
            "rvalue {other:?} is not in the current generic-host cohort"
        ))),
    }
}

fn execute_embedded_statement(
    state: &mut HostState,
    statement: &runmat_mir::MirStmtKind,
) -> JitResult<()> {
    match statement {
        runmat_mir::MirStmtKind::Assign {
            place: runmat_mir::MirPlace::Local(local),
            value,
        } => {
            let mut values = evaluate_rvalue(state, value)?;
            if values.len() != 1 {
                return Err(JitError::Host(
                    "embedded short-circuit assignment did not produce one value".into(),
                ));
            }
            let slot = state.locals.get_mut(local.0).ok_or_else(|| {
                JitError::Host("embedded assignment local is out of bounds".into())
            })?;
            *slot = values.remove(0);
            Ok(())
        }
        runmat_mir::MirStmtKind::Expr(value) => {
            let _ = evaluate_rvalue(state, value)?;
            Ok(())
        }
        other => Err(JitError::UnsupportedSite(format!(
            "embedded short-circuit statement {other:?} is not supported"
        ))),
    }
}

fn logical_truth(state: &HostState, value: &Value, label: &str) -> JitResult<bool> {
    futures::executor::block_on(state.runtime.scope(
        runmat_runtime::condition::logical_truth_from_value(value, label),
    ))
    .map_err(JitError::from)
}

pub(super) fn materialize_operand(state: &mut HostState, operand: &MirOperand) -> JitResult<Value> {
    let reference = evaluate_operand(state, operand)?;
    state.arena.get(reference).cloned()
}

pub(super) fn evaluate_operand(
    state: &mut HostState,
    operand: &MirOperand,
) -> JitResult<NativeValueRef> {
    match operand {
        MirOperand::Local(local) => state
            .locals
            .get(local.0)
            .copied()
            .ok_or_else(|| JitError::Host(format!("local {} is out of bounds", local.0))),
        MirOperand::Constant(constant) => {
            let value = match constant {
                MirConstant::Number(text) => Value::Num(text.parse::<f64>().map_err(|error| {
                    JitError::Host(format!("invalid MIR numeric constant {text:?}: {error}"))
                })?),
                MirConstant::String(text) => Value::String(text.0.clone()),
                MirConstant::Symbol(symbol) => Value::String(symbol.0.clone()),
                MirConstant::Bool(value) => Value::Bool(*value),
                MirConstant::EmptyArray => Value::Tensor(
                    runmat_value::Tensor::new(Vec::new(), vec![0, 0]).map_err(JitError::Host)?,
                ),
            };
            Ok(state.arena.insert(value))
        }
        MirOperand::FunctionHandle(identity) => identity
            .display_name()
            .map(Value::FunctionHandle)
            .map(|value| state.arena.insert(value))
            .ok_or_else(|| {
                JitError::UnsupportedSite(format!(
                    "bound function handle {identity:?} requires R14 closure state"
                ))
            }),
    }
}
