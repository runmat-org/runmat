//! Original compiled-model authority is additional to ordinary QP checks.
use super::*;
pub(in crate::builtins::math::optim) use crate::builtins::math::optim::linprog::compiled::CertificateCandidate;
use conic::LinearRowRole;
use num_rational::BigRational;
pub(in crate::builtins::math::optim) enum RestrictedCurvature {
    /// Public supplied matrices are the original model.
    Supplied,
    /// Independent original source has certified restricted convexity.
    Convex,
    Nonconvex,
}
mod warm;
pub(in crate::builtins::math::optim::quadprog) use warm::WarmSupplied;

/// Acceptance of the original model, never a replacement for QP PSD/KKT checks.
pub(in crate::builtins::math::optim) trait Validation {
    fn saved_working_set(&self) -> Option<&runmat_value::warm_start::LinearWorkingSetData> {
        None
    }
    fn wants_working_set(&self) -> bool {
        false
    }
    fn retain_working_set(&mut self, _data: runmat_value::warm_start::LinearWorkingSetData) {}
    async fn stopped_candidate(
        &mut self,
        point: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool>;
    async fn objective_limit(
        &mut self,
        point: &[f64],
        cost: f64,
        limit: f64,
        feasibility: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<bool>>;
    /// Additional exact original-source curvature decision, independent of
    /// the existing supplied binary64 matrix certificate.
    fn original_convexity(&self) -> bool;
    /// Active-set curvature on the actual original equality affine space.
    async fn restricted_convexity(
        &mut self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<RestrictedCurvature>;
    /// Additional source proof for a feasible anchor and projected-flat ray.
    async fn recession(
        &mut self,
        point: &[BigRational],
        direction: &[BigRational],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool>;
    async fn solution(
        &mut self,
        point: &[f64],
        roles: &[LinearRowRole],
        duals: &[f64],
        feasibility: f64,
        optimality: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool>;
    async fn incumbent(
        &mut self,
        point: &[f64],
        feasibility: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool>;
    async fn certificate(
        &mut self,
        candidate: CertificateCandidate<'_>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool>;
}
/// Public quadprog's supplied matrices are its original model.
pub(in crate::builtins::math::optim::quadprog) struct SuppliedModel;
impl Validation for SuppliedModel {
    async fn stopped_candidate(
        &mut self,
        _: &[f64],
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Ok(true)
    }
    async fn objective_limit(
        &mut self,
        _: &[f64],
        cost: f64,
        limit: f64,
        _: f64,
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<bool>> {
        Ok(Some(cost < limit))
    }
    fn original_convexity(&self) -> bool {
        true
    }
    async fn restricted_convexity(
        &mut self,
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<RestrictedCurvature> {
        Ok(RestrictedCurvature::Supplied)
    }
    async fn recession(
        &mut self,
        _: &[BigRational],
        _: &[BigRational],
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Ok(true)
    }
    async fn solution(
        &mut self,
        _: &[f64],
        _: &[LinearRowRole],
        _: &[f64],
        _: f64,
        _: f64,
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Ok(true)
    }
    async fn incumbent(
        &mut self,
        _: &[f64],
        _: f64,
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
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
pub(in crate::builtins::math::optim::quadprog) struct Terminal<'a, V> {
    pub roles: &'a [LinearRowRole],
    pub validation: &'a mut V,
}
impl<V: Validation> conic::solve::TerminalValidation for Terminal<'_, V> {
    async fn farkas(
        &mut self,
        weights: &[BigRational],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.validation
            .certificate(
                CertificateCandidate::Farkas {
                    rows: self.roles,
                    multipliers: weights,
                },
                cooperation,
            )
            .await
    }
    async fn ray(
        &mut self,
        direction: &[BigRational],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.validation
            .certificate(CertificateCandidate::Descent(direction), cooperation)
            .await
    }
    async fn feasible(
        &mut self,
        point: &[BigRational],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.validation
            .certificate(CertificateCandidate::Feasible(point), cooperation)
            .await
    }
}
