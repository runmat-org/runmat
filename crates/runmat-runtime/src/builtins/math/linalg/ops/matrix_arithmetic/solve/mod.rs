mod errors;
mod host;
mod integer;
mod provider;

use runmat_value::{Tensor, Value};

use crate::BuiltinResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SolveOrientation {
    Left,
    Right,
}

impl SolveOrientation {
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Left => "mldivide",
            Self::Right => "mrdivide",
        }
    }
}

pub(super) async fn evaluate(
    orientation: SolveOrientation,
    lhs: &Value,
    rhs: &Value,
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
    let lhs = crate::dispatcher::gather_if_needed_async(lhs)
        .await
        .map_err(|error| errors::map_control_flow(orientation, error))?;
    let rhs = crate::dispatcher::gather_if_needed_async(rhs)
        .await
        .map_err(|error| errors::map_control_flow(orientation, error))?;
    host::evaluate(orientation, lhs, rhs)
}

pub(super) fn host_real(
    orientation: SolveOrientation,
    lhs: &Tensor,
    rhs: &Tensor,
) -> BuiltinResult<Tensor> {
    host::real(orientation, lhs, rhs)
}
