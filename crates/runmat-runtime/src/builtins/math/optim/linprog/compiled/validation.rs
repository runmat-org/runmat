//! Domain evidence supplements, never replaces, the canonical LP validators.
pub(in crate::builtins::math::optim) use super::super::model::RowRole;
use super::*;
use num_rational::BigRational;
use std::future::Future;
pub(in crate::builtins::math::optim) enum CertificateCandidate<'a> {
    InconsistentBounds {
        column: usize,
        lower: f64,
        upper: f64,
    },
    Farkas {
        rows: &'a [RowRole],
        multipliers: &'a [BigRational],
    },
    Descent(&'a [BigRational]),
    Feasible(&'a [BigRational]),
}
pub(in crate::builtins::math::optim) trait Validation {
    fn solution<'a>(
        &'a mut self,
        point: &'a [f64],
        rows: &'a [RowRole],
        duals: &'a [f64],
        cooperation: &'a mut ScalarCooperation,
    ) -> impl Future<Output = BuiltinResult<bool>> + 'a;
    fn incumbent<'a>(
        &'a mut self,
        point: &'a [f64],
        cooperation: &'a mut ScalarCooperation,
    ) -> impl Future<Output = BuiltinResult<bool>> + 'a;
    fn certificate<'a>(
        &'a mut self,
        candidate: CertificateCandidate<'a>,
        cooperation: &'a mut ScalarCooperation,
    ) -> impl Future<Output = BuiltinResult<bool>> + 'a;
}
/// Public linprog has no additional domain beyond its supplied, already
/// checked original LP rows. It still runs every canonical primal/KKT/exact
/// certificate check; a compiled-model consumer must provide its own validator.
pub(in crate::builtins::math::optim::linprog) struct SuppliedRows;
impl Validation for SuppliedRows {
    async fn solution(
        &mut self,
        _: &[f64],
        _: &[RowRole],
        _: &[f64],
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Ok(true)
    }
    async fn incumbent(&mut self, _: &[f64], _: &mut ScalarCooperation) -> BuiltinResult<bool> {
        Ok(true)
    }
    async fn certificate(
        &mut self,
        _: CertificateCandidate<'_>,
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Ok(true)
    }
}
pub(in crate::builtins::math::optim::linprog) async fn farkas<V: Validation>(
    model: &model::ConicModel,
    z: &[f64],
    cooperation: &mut ScalarCooperation,
    validation: &mut V,
) -> BuiltinResult<bool> {
    let cones = model.cones();
    let view = certificates::conic_view(model, &cones);
    if model.roles.len() != model.b.len() {
        return Ok(false);
    }
    crate::builtins::math::optim::common::conic::terminal::primal_infeasible_satisfying(
        &view,
        z,
        cooperation,
        async |candidate: &[BigRational], cooperation: &mut ScalarCooperation| {
            validation
                .certificate(
                    CertificateCandidate::Farkas {
                        rows: &model.roles,
                        multipliers: candidate,
                    },
                    cooperation,
                )
                .await
        },
    )
    .await
}
pub(in crate::builtins::math::optim::linprog) async fn descent<V: Validation>(
    model: &model::ConicModel,
    direction: &[f64],
    cooperation: &mut ScalarCooperation,
    validation: &mut V,
) -> BuiltinResult<bool> {
    let cones = model.cones();
    let view = certificates::conic_view(model, &cones);
    Ok(
        crate::builtins::math::optim::common::conic::terminal::verified_descent_ray_satisfying(
            &view,
            direction,
            cooperation,
            async |candidate: &[BigRational], cooperation: &mut ScalarCooperation| {
                validation
                    .certificate(CertificateCandidate::Descent(candidate), cooperation)
                    .await
            },
        )
        .await?
        .is_some(),
    )
}
pub(in crate::builtins::math::optim::linprog) async fn feasible<V: Validation>(
    model: &model::ConicModel,
    point: &[f64],
    cooperation: &mut ScalarCooperation,
    validation: &mut V,
) -> BuiltinResult<bool> {
    let cones = model.cones();
    let view = certificates::conic_view(model, &cones);
    Ok(
        crate::builtins::math::optim::common::conic::terminal::verified_point_satisfying(
            &view,
            point,
            cooperation,
            async |candidate: &[BigRational], cooperation: &mut ScalarCooperation| {
                validation
                    .certificate(CertificateCandidate::Feasible(candidate), cooperation)
                    .await
            },
        )
        .await?
        .is_some(),
    )
}
