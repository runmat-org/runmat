//! Shared unary arithmetic semantics for bytecode and native execution.

use runmat_value::Value;

use crate::BuiltinResult;

#[derive(Clone, Copy, Debug)]
pub enum UnaryArithmeticOperation {
    Plus,
    Minus,
}

pub async fn evaluate(operation: UnaryArithmeticOperation, value: Value) -> BuiltinResult<Value> {
    evaluate_in_context(None, operation, value).await
}

pub async fn evaluate_in_context(
    context: Option<&crate::context::RuntimeContext>,
    operation: UnaryArithmeticOperation,
    value: Value,
) -> BuiltinResult<Value> {
    if matches!(&value, Value::ComplexTensor(tensor) if tensor.integer_storage().is_some()) {
        return Err(crate::runtime_error::semantic_error(
            "ComplexIntegerArithmetic",
            "complex integer arithmetic is not supported",
        ));
    }

    let operator = match operation {
        UnaryArithmeticOperation::Plus => runmat_types::OperatorKind::UnaryPlus,
        UnaryArithmeticOperation::Minus => runmat_types::OperatorKind::UnaryMinus,
    };
    let method = operator
        .overload_name()
        .expect("unary arithmetic has an overload method");
    // Resolve registered object/handle methods before numeric fallback. Absence
    // permits fallback; a method's own error must propagate without replay.
    if let Some(result) = crate::dispatcher::try_call_registered_instance_method(
        method.display_name(),
        std::slice::from_ref(&value),
        Some(1),
        context,
    )
    .await?
    {
        return crate::dispatcher::compatibility_checked_builtin_result(
            method.display_name(),
            std::slice::from_ref(&value),
            result,
        );
    }

    match operation {
        UnaryArithmeticOperation::Plus => Ok(value),
        UnaryArithmeticOperation::Minus => match context {
            Some(context) => {
                crate::call_builtin_async_in_context(context, "times", &[value, Value::Num(-1.0)])
                    .await
            }
            None => crate::call_builtin_async("times", &[value, Value::Num(-1.0)]).await,
        },
    }
}

#[cfg(test)]
mod tests;
