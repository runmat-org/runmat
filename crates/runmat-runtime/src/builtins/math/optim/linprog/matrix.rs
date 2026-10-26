//! LP's exact-integer policy at the shared canonical CSC boundary.
use super::*;
use crate::builtins::math::optim::common::conic::matrix::{self, IntegerPolicy};
pub(super) type MatrixInput = clarabel::algebra::CscMatrix<f64>;
pub(super) async fn parse(
    label: &str,
    value: Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<MatrixInput>> {
    matrix::parse(
        gather(value).await?,
        IntegerPolicy::Exact,
        cooperation,
        |detail| {
            linprog_error_with_detail(&LINPROG_ERROR_INVALID_INPUT, format!("{label} {detail}"))
        },
    )
    .await
}
