use runmat_runtime::native::NativeValueRef;
use runmat_types::OperatorKind;
use runmat_value::Value;

use crate::JitResult;

use super::state::HostState;

pub(super) fn evaluate(
    state: &mut HostState,
    operator: OperatorKind,
    arguments: Vec<Value>,
) -> JitResult<NativeValueRef> {
    let name = builtin_name(operator);
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

fn builtin_name(operator: OperatorKind) -> &'static str {
    match operator {
        OperatorKind::UnaryPlus => "uplus",
        OperatorKind::UnaryMinus => "uminus",
        OperatorKind::Not => "not",
        OperatorKind::Add => "plus",
        OperatorKind::Subtract => "minus",
        OperatorKind::MatrixMultiply => "mtimes",
        OperatorKind::ElementwiseMultiply => "times",
        OperatorKind::MatrixPower => "mpower",
        OperatorKind::ElementwisePower => "power",
        OperatorKind::Mldivide => "mldivide",
        OperatorKind::Mrdivide => "mrdivide",
        OperatorKind::ElementwiseDivide => "rdivide",
        OperatorKind::ElementwiseLeftDivide => "ldivide",
        OperatorKind::Equal => "eq",
        OperatorKind::NotEqual => "ne",
        OperatorKind::Less => "lt",
        OperatorKind::LessEqual => "le",
        OperatorKind::Greater => "gt",
        OperatorKind::GreaterEqual => "ge",
        OperatorKind::ElementwiseAnd => "and",
        OperatorKind::ElementwiseOr => "or",
        OperatorKind::Transpose => "transpose",
        OperatorKind::ConjugateTranspose => "ctranspose",
        OperatorKind::ShortCircuitAnd | OperatorKind::ShortCircuitOr => {
            unreachable!("short-circuit operators are represented by MirRvalue::ShortCircuit")
        }
    }
}
