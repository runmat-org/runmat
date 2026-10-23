mod errors;
mod host;
mod integer;
mod orientation;
mod provider;

use runmat_value::Value;

use crate::context::RuntimeContext;
use crate::BuiltinResult;

pub(super) use host::real as host_real;
pub(in crate::builtins::math::linalg) use orientation::SolveOrientation;

pub(in crate::builtins::math::linalg) async fn evaluate(
    orientation: SolveOrientation,
    lhs: &Value,
    rhs: &Value,
) -> BuiltinResult<Value> {
    evaluate_with_context(orientation, lhs, rhs, None).await
}

pub(in crate::builtins::math::linalg) async fn evaluate_with_context(
    orientation: SolveOrientation,
    lhs: &Value,
    rhs: &Value,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::is_typed_complex_integer(lhs)
        || crate::builtins::common::validation::is_typed_complex_integer(rhs)
    {
        return Err(errors::invalid_input(
            orientation,
            "complex integer arithmetic is not supported",
        ));
    }
    if integer::contains(lhs) || integer::contains(rhs) {
        return integer::evaluate(orientation, lhs, rhs).await;
    }
    if let Some(result) = provider::try_solve(orientation, lhs, rhs).await? {
        return Ok(result);
    }
    // This new host route requires the actual execution policy issuer. Legacy
    // context-free callers retain their prior sparse behavior.
    if let (SolveOrientation::Left, Some(context)) = (orientation, context) {
        if let Some(result) = host::try_sparse_triangular(lhs, rhs, context).await? {
            return Ok(result);
        }
        if let Some(result) = host::try_general_sparse(lhs, rhs, context).await? {
            return Ok(result);
        }
        if let Some(result) = host::try_rectangular_sparse(lhs, rhs, context).await? {
            return Ok(result);
        }
    }
    let lhs = crate::dispatcher::gather_if_needed_async(lhs)
        .await
        .map_err(|error| errors::map_control_flow(orientation, error))?;
    let rhs = crate::dispatcher::gather_if_needed_async(rhs)
        .await
        .map_err(|error| errors::map_control_flow(orientation, error))?;
    host::evaluate(orientation, lhs, rhs)
}
