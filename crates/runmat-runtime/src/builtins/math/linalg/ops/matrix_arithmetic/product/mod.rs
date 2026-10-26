//! Shared matrix-product evaluation for scalar and matrix/page consumers.
//! Public identities own admission and error descriptors; numerical/provider
//! semantics and temporary ownership have one implementation here.
use runmat_value::Value;
// Domain failures are returned inside the canonical operational error channel.
// This early-return syntax moves neither diagnostics nor values into a box.
macro_rules! product_try {
    ($result:expr) => {
        match $result {
            Ok(value) => value,
            Err(error) => return Ok(Err(error)),
        }
    };
}
mod errors;
mod host;
mod integer;
mod provider;
mod sparse;
pub(crate) use errors::ProductError;
type ProductResult<T> = Result<T, ProductError>;
// Mathematical operation label retained for established helper error messages.
const NAME: &str = "mtimes";
pub(crate) async fn evaluate(
    lhs: &Value,
    rhs: &Value,
) -> crate::BuiltinResult<ProductResult<Value>> {
    evaluate_in_context(None, lhs, rhs).await
}

pub(crate) async fn evaluate_in_context(
    context: Option<&crate::context::RuntimeContext>,
    lhs: &Value,
    rhs: &Value,
) -> crate::BuiltinResult<ProductResult<Value>> {
    if integer::contains(lhs) || integer::contains(rhs) {
        return integer::evaluate(context, lhs, rhs).await;
    }
    if let Some(result) = product_try!(provider::try_product(lhs, rhs).await?) {
        return Ok(Ok(result));
    }
    let route = context.and_then(|runtime| {
        let route = runtime.vm_fusion_host_route()?;
        let scalar = |value: &Value| {
            matches!(value, Value::Num(_))
                || matches!(value, Value::Tensor(tensor) if tensor.len() == 1)
        };
        (crate::builtins::math::host_route::real_floating_host(lhs)
            && crate::builtins::math::host_route::real_floating_host(rhs)
            && (scalar(lhs) || scalar(rhs))
            && route.allows_intrinsic_host(runtime, runmat_types::OperatorKind::MatrixMultiply))
        .then(|| route.clone())
    });
    if let (Some(runtime), Some(route)) = (context, route.as_ref()) {
        if let Some(output) = route
            .take_intrinsic_numerical_output(runtime, runmat_types::OperatorKind::MatrixMultiply)?
        {
            let result = crate::builtins::math::elementwise::binary_arithmetic::preparation::managed::execute_value(
                runtime,
                output,
                runmat_types::OperatorKind::ElementwiseMultiply,
                lhs,
                rhs,
            )
            ?;
            route.confirm_intrinsic_host(runtime, runmat_types::OperatorKind::MatrixMultiply);
            return Ok(Ok(result));
        }
    }
    let output = product_try!(host::evaluate(context, lhs.clone(), rhs.clone()).await?);
    if let Some(route) = route {
        route.confirm_intrinsic_host(context.unwrap(), runmat_types::OperatorKind::MatrixMultiply);
    }
    Ok(Ok(output))
}
