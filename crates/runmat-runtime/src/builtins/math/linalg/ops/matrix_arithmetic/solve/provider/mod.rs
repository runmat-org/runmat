mod admission;
mod execution;
mod operand;
mod shape;

use runmat_value::Value;

use crate::BuiltinResult;

use super::SolveOrientation;
use admission::{is_complex, is_resident, select_provider, selected_precision};
use execution::invoke;
use operand::PreparedOperand;
use shape::{disallowed_scalar, output_shape};

pub(super) async fn try_solve(
    orientation: SolveOrientation,
    lhs: &Value,
    rhs: &Value,
) -> BuiltinResult<Option<Value>> {
    if !is_resident(lhs) && !is_resident(rhs) {
        return Ok(None);
    }
    if is_complex(lhs) || is_complex(rhs) {
        return Ok(None);
    }
    let Some(provider) = select_provider(orientation, lhs, rhs)? else {
        return Ok(None);
    };
    if selected_precision(lhs, rhs) != Some(provider.precision()) {
        return Ok(None);
    }

    let Some(lhs) = PreparedOperand::new(orientation, lhs, provider)? else {
        return Ok(None);
    };
    let Some(rhs) = PreparedOperand::new(orientation, rhs, provider)? else {
        return Ok(None);
    };
    if disallowed_scalar(orientation, lhs.handle(), rhs.handle()) {
        return Ok(None);
    }
    invoke(
        orientation,
        provider,
        lhs.handle(),
        rhs.handle(),
        output_shape(orientation, lhs.handle(), rhs.handle()),
    )
    .await
}
