//! QP's original-DAG authority supplements, never replaces, numerical QP KKT.
use super::*;
use crate::builtins::math::optim::common::conic::LinearRowRole;
use crate::builtins::math::optim::quadprog::compiled as qp;
impl qp::Validation for Validator<'_> {
    async fn stopped_candidate(
        &mut self,
        point: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.stopped_original(point, c).await
    }
    async fn objective_limit(
        &mut self,
        point: &[f64],
        _: f64,
        limit: f64,
        feasibility: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<bool>> {
        self.objective_limit_original(point, limit, feasibility, c)
            .await
    }
    async fn restricted_convexity(
        &mut self,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<qp::RestrictedCurvature> {
        Ok(if self.authority.restricted_convexity(c).await? {
            qp::RestrictedCurvature::Convex
        } else {
            qp::RestrictedCurvature::Nonconvex
        })
    }
    async fn recession(
        &mut self,
        point: &[num_rational::BigRational],
        direction: &[num_rational::BigRational],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.authority
            .quadratic_recession(point, direction, c)
            .await
    }
    fn original_convexity(&self) -> bool {
        self.program.objective.quadratic().is_some_and(|evidence| {
            evidence.curvature == super::super::quadratic::Curvature::Convex
        })
    }
    async fn solution(
        &mut self,
        point: &[f64],
        roles: &[LinearRowRole],
        duals: &[f64],
        feasibility: f64,
        optimality: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if !self.prepare(point, c).await?
            || !self.primal_at_tolerance(point, feasibility, c).await?
        {
            return Ok(false);
        }
        self.authority
            .kkt_at_tolerance(point, roles, duals, optimality, c)
            .await
    }
    async fn incumbent(
        &mut self,
        point: &[f64],
        feasibility: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if !self.prepare(point, c).await? {
            return Ok(false);
        }
        self.primal_at_tolerance(point, feasibility, c).await
    }
    async fn certificate(
        &mut self,
        candidate: qp::CertificateCandidate<'_>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.authority.certificate(candidate, c).await
    }
}
