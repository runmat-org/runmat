//! One original point/evaluation cache and exact original terminal authority
//! supplement the existing LP numerical validators without another solver.
use super::super::super::linprog::compiled::{CertificateCandidate, RowRole, Validation};
use super::*;
mod integer;
mod least_squares;
mod primal;
mod quadratic;
pub(super) mod stopped;
pub(super) struct Validator<'a> {
    authority: proof::Authority<'a>,
    program: &'a Program,
    formulation: &'a formulation::Formulation,
    point: Option<numerical::Point>,
    stopped: Option<stopped::Candidate>,
    integer_rows: Option<integer::Rows>,
    policy: Policy,
}
impl<'a> Validator<'a> {
    pub(super) async fn new(
        program: &'a Program,
        formulation: &'a formulation::Formulation,
        rows: &[assembly::Rows],
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        Ok(Self {
            authority: proof::Authority::new(program, formulation, rows, policy, cooperation)
                .await?,
            program,
            formulation,
            point: None,
            stopped: None,
            integer_rows: None,
            policy,
        })
    }
    async fn prepare(
        &mut self,
        candidate: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if let Some(point) = &self.point {
            if point
                .matches(self.program, candidate, self.policy, cooperation)
                .await?
            {
                return Ok(true);
            }
        }
        self.point = numerical::Point::evaluate(
            self.program,
            &self.formulation.roots,
            candidate,
            self.policy,
            cooperation,
        )
        .await?;
        Ok(self.point.is_some())
    }
    pub(super) fn into_stopped(self) -> (Option<numerical::Point>, Option<stopped::Candidate>) {
        (self.point, self.stopped)
    }
    pub(super) fn into_point(self) -> Option<numerical::Point> {
        self.point
    }
}
impl Validation for Validator<'_> {
    async fn solution(
        &mut self,
        point: &[f64],
        rows: &[RowRole],
        duals: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if !self.prepare(point, cooperation).await? || !self.primal(point, cooperation).await? {
            return Ok(false);
        }
        self.authority.kkt(point, rows, duals, cooperation).await
    }
    async fn incumbent(
        &mut self,
        point: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if !self.prepare(point, cooperation).await? {
            return Ok(false);
        }
        self.primal(point, cooperation).await
    }
    async fn certificate(
        &mut self,
        candidate: CertificateCandidate<'_>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.authority.certificate(candidate, cooperation).await
    }
}
