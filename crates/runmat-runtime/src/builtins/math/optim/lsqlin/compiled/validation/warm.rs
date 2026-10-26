//! Raw numerical admission plus immutable algorithm state; no proof upgrade.
use super::*;
use runmat_value::warm_start::LinearWorkingSetData;
pub(in crate::builtins::math::optim::lsqlin) struct WarmSupplied<'a> {
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
    async fn stopped_candidate(
        &mut self,
        x: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Supplied.stopped_candidate(x, c).await
    }
    async fn objective_limit(
        &mut self,
        x: &[f64],
        cost: f64,
        limit: f64,
        feasibility: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<bool>> {
        Supplied
            .objective_limit(x, cost, limit, feasibility, c)
            .await
    }
    async fn solution(
        &mut self,
        x: &[f64],
        rows: Rows<'_>,
        z: &[f64],
        feasibility: f64,
        optimality: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Supplied
            .solution(x, rows, z, feasibility, optimality, c)
            .await
    }
    async fn incumbent(
        &mut self,
        x: &[f64],
        feasibility: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Supplied.incumbent(x, feasibility, c).await
    }
    async fn certificate(
        &mut self,
        candidate: Certificate<'_>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Supplied.certificate(candidate, c).await
    }
}
