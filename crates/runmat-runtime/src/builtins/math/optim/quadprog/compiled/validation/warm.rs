//! Current supplied QP proof plus immutable saved/completed row data.
use super::*;
use runmat_value::warm_start::LinearWorkingSetData;
pub(in crate::builtins::math::optim::quadprog) struct WarmSupplied<'a> {
    pub saved: Option<&'a LinearWorkingSetData>,
    pub completed: Option<LinearWorkingSetData>,
}
impl Validation for WarmSupplied<'_> {
    fn saved_working_set(&self) -> Option<&LinearWorkingSetData> {
        self.saved
    }
    fn wants_working_set(&self) -> bool {
        true
    }
    fn retain_working_set(&mut self, data: LinearWorkingSetData) {
        self.completed = Some(data);
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
    async fn stopped_candidate(
        &mut self,
        point: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        SuppliedModel.stopped_candidate(point, c).await
    }
    async fn objective_limit(
        &mut self,
        point: &[f64],
        cost: f64,
        limit: f64,
        tolerance: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<bool>> {
        SuppliedModel
            .objective_limit(point, cost, limit, tolerance, c)
            .await
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
        SuppliedModel
            .solution(point, roles, duals, feasibility, optimality, c)
            .await
    }
    async fn incumbent(
        &mut self,
        point: &[f64],
        feasibility: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        SuppliedModel.incumbent(point, feasibility, c).await
    }
    async fn certificate(
        &mut self,
        candidate: CertificateCandidate<'_>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        SuppliedModel.certificate(candidate, c).await
    }
}
