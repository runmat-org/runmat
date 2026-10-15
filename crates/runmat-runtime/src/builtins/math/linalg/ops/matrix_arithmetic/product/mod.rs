//! Shared matrix-product evaluation for scalar and matrix/page consumers.
//! Public identities own admission and error descriptors; numerical/provider
//! semantics and temporary ownership have one implementation here.
use runmat_value::Value;
mod errors;
mod host;
mod integer;
mod provider;
pub(crate) use errors::ProductError;
type ProductResult<T> = Result<T, ProductError>;
// Mathematical operation label retained for established helper error messages.
const NAME: &str = "mtimes";
pub(crate) async fn evaluate(lhs: &Value, rhs: &Value) -> ProductResult<Value> {
    evaluate_in_context(None, lhs, rhs).await
}

pub(crate) async fn evaluate_in_context(
    context: Option<&crate::context::RuntimeContext>,
    lhs: &Value,
    rhs: &Value,
) -> ProductResult<Value> {
    if integer::contains(lhs) || integer::contains(rhs) {
        return integer::evaluate(lhs, rhs).await;
    }
    if let Some(result) = provider::try_product(lhs, rhs).await? {
        return Ok(result);
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
    let output = host::evaluate(lhs.clone(), rhs.clone()).await?;
    if let Some(route) = route {
        route.confirm_intrinsic_host(context.unwrap(), runmat_types::OperatorKind::MatrixMultiply);
    }
    Ok(output)
}
