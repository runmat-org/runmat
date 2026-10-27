//! Shared equation qualification of a residual-controller terminal.
use super::*;
pub(in crate::builtins::math::optim) async fn require_root(
    result: &mut LeastSquaresResult,
    tolerance: f64,
    cooperation: Option<&mut ScalarCooperation>,
) -> BuiltinResult<()> {
    if result.exitflag > 0 && residual_norm_inf(&result.residual, cooperation).await? > tolerance {
        result.exitflag = -2;
        result.message =
            "Equation not solved. The solver converged to a point with a nonzero residual."
                .to_string();
    }
    Ok(())
}
