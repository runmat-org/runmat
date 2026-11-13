//! Additional original-domain checks participate in EACH exact recovery
//! candidate, never after a rounded-model status has discarded its evidence.
use super::*;
use num_rational::BigRational;
pub(crate) trait TerminalValidation {
    async fn farkas(
        &mut self,
        weights: &[BigRational],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool>;
    async fn ray(
        &mut self,
        direction: &[BigRational],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool>;
    async fn feasible(
        &mut self,
        point: &[BigRational],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool>;
}
pub(crate) struct SuppliedModel;
impl TerminalValidation for SuppliedModel {
    async fn farkas(
        &mut self,
        _: &[BigRational],
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Ok(true)
    }
    async fn ray(&mut self, _: &[BigRational], _: &mut ScalarCooperation) -> BuiltinResult<bool> {
        Ok(true)
    }
    async fn feasible(
        &mut self,
        _: &[BigRational],
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Ok(true)
    }
}
pub(super) async fn infeasible<V: TerminalValidation>(
    model: &Model<'_>,
    weights: &[f64],
    c: &mut ScalarCooperation,
    validation: &mut V,
) -> BuiltinResult<bool> {
    terminal::primal_infeasible_satisfying(
        model,
        weights,
        c,
        async |candidate: &[BigRational], c: &mut ScalarCooperation| {
            validation.farkas(candidate, c).await
        },
    )
    .await
}
pub(super) async fn ray<V: TerminalValidation>(
    model: &Model<'_>,
    direction: &[f64],
    c: &mut ScalarCooperation,
    validation: &mut V,
) -> BuiltinResult<Option<terminal::VerifiedRay>> {
    terminal::verified_descent_ray_satisfying(
        model,
        direction,
        c,
        async |candidate: &[BigRational], c: &mut ScalarCooperation| {
            validation.ray(candidate, c).await
        },
    )
    .await
}
pub(super) async fn feasible<V: TerminalValidation>(
    model: &Model<'_>,
    point: &[f64],
    c: &mut ScalarCooperation,
    validation: &mut V,
) -> BuiltinResult<Option<terminal::VerifiedPoint>> {
    terminal::verified_point_satisfying(
        model,
        point,
        c,
        async |candidate: &[BigRational], c: &mut ScalarCooperation| {
            validation.feasible(candidate, c).await
        },
    )
    .await
}
