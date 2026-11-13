//! Shared unary arithmetic semantics for bytecode and native execution.

use runmat_value::Value;

use crate::execution::RootedValue;
use crate::BuiltinResult;

#[derive(Clone, Copy, Debug)]
pub enum UnaryArithmeticOperation {
    Plus,
    Minus,
}

pub async fn evaluate(
    operation: UnaryArithmeticOperation,
    value: Value,
) -> BuiltinResult<RootedValue> {
    evaluate_in_context(None, operation, value).await
}

pub async fn evaluate_in_context(
    context: Option<&crate::context::RuntimeContext>,
    operation: UnaryArithmeticOperation,
    value: Value,
) -> BuiltinResult<RootedValue> {
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
    if let Some((runtime, route)) = context.and_then(|runtime| {
        runtime
            .native_region_host_route()
            .map(|route| (runtime, route))
    }) {
        route.require_intrinsic_host(runtime, operator)?;
        if !crate::builtins::math::host_route::real_floating_host(&value) {
            return Err(crate::runtime_error::semantic_error(
                "RunMat:Placement:InvalidNativeRegion",
                "native host intrinsic received a non-host operand",
            ));
        }
    }
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
        UnaryArithmeticOperation::Plus => {
            if crate::builtins::math::host_route::real_floating_host(&value) {
                if let Some(context) = context {
                    if let Some(route) =
                        crate::builtins::math::host_route::intrinsic(context, operator)
                    {
                        route.confirm_intrinsic_host(context, operator);
                    }
                }
            }
            RootedValue::new(value).map_err(|error| error.into_runtime_error())
        }
        UnaryArithmeticOperation::Minus => {
            if crate::builtins::math::host_route::real_floating_host(&value) {
                if let Some((context, route)) = context.and_then(|runtime| {
                    crate::builtins::math::host_route::intrinsic(runtime, operator)
                        .map(|route| (runtime, route))
                }) {
                    if let Some(output) =
                        route.take_intrinsic_numerical_output(context, operator)?
                    {
                        let result = crate::builtins::math::elementwise::binary_arithmetic::preparation::managed::execute_value(
                            context,
                            output,
                            runmat_types::OperatorKind::ElementwiseMultiply,
                            &Value::Num(-1.0),
                            &value,
                        )?;
                        route.confirm_intrinsic_host(context, operator);
                        return RootedValue::new(result)
                            .map_err(|error| error.into_runtime_error());
                    }
                }
            }
            match context {
                Some(context) => {
                    crate::call_builtin_async_in_context(
                        context,
                        "times",
                        &[value, Value::Num(-1.0)],
                    )
                    .await
                }
                None => crate::call_builtin_async("times", &[value, Value::Num(-1.0)]).await,
            }
        }
    }
}

#[cfg(test)]
mod tests;
