use runmat_mir::{MirConstant, MirOperand, MirRvalue};
use runmat_runtime::native::NativeValueRef;
use runmat_value::Value;

use crate::{JitError, JitResult};

use super::state::HostState;

pub(super) fn evaluate_rvalue(
    state: &mut HostState,
    value: &MirRvalue,
) -> JitResult<NativeValueRef> {
    match value {
        MirRvalue::Use(operand) => evaluate_operand(state, operand),
        other => Err(JitError::UnsupportedSite(format!(
            "rvalue {other:?} is not in the first generic-host cohort"
        ))),
    }
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
