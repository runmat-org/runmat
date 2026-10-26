//! Evaluated-only algebraic stop authority, never a feasible/KKT cache entry.
use super::*;
impl Validator<'_> {
    pub(super) async fn stopped_original(
        &mut self,
        point: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if let Some(cached) = self.stopped.as_ref() {
            if cached
                .matches(&self.program.layout, point, self.policy, c)
                .await?
            {
                return Ok(true);
            }
        }
        if let Some(cached) = &self.point {
            if cached.matches(self.program, point, self.policy, c).await? {
                self.stopped = Some(stopped::Candidate::from_point(
                    self.point.take().expect("matching original point"),
                    self.formulation,
                    self.policy,
                )?);
                return Ok(true);
            }
        }
        self.stopped =
            stopped::Candidate::evaluate(self.program, self.formulation, point, self.policy, c)
                .await?;
        Ok(self.stopped.is_some())
    }
    pub(super) async fn objective_limit_original(
        &mut self,
        point: &[f64],
        limit: f64,
        feasibility: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<bool>> {
        let Some(evaluated) = numerical::Point::evaluate(
            self.program,
            &self.formulation.roots,
            point,
            self.policy,
            c,
        )
        .await?
        else {
            return Ok(None);
        };
        if !self
            .primal_measured(&evaluated, point, feasibility, c)
            .await?
        {
            // An evaluated-only or infeasible probe is not put in point.
            self.stopped = Some(stopped::Candidate::from_point(
                evaluated,
                self.formulation,
                self.policy,
            )?);
            return Ok(Some(false));
        }
        let value = match &self.formulation.objective {
            Some(objective) => evaluated.at(objective.root, 0, self.policy)?,
            None => 0.,
        };
        if !value.is_finite() {
            return Ok(None);
        }
        let value = if self
            .formulation
            .objective
            .as_ref()
            .is_some_and(|objective| {
                objective.sense
                    == runmat_value::optimization_model::OptimizationObjectiveSense::Maximize
            }) {
            -value
        } else {
            value
        };
        self.point = Some(evaluated);
        Ok(Some(value < limit))
    }
}
pub(in crate::builtins::math::optim::optimmodel::lowering) struct Candidate {
    solution: RootedValue,
    objective: Option<f64>,
}
impl Candidate {
    pub(in crate::builtins::math::optim::optimmodel::lowering) fn solution(&self) -> &RootedValue {
        &self.solution
    }
    pub(in crate::builtins::math::optim::optimmodel::lowering) fn objective(&self) -> Option<f64> {
        self.objective
    }
    pub(in crate::builtins::math::optim::optimmodel::lowering) async fn supplied(
        layout: &layout::Layout,
        coordinates: &[f64],
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        Ok(numerical::solution(layout, coordinates, policy, c)
            .await?
            .map(|solution| Self {
                solution,
                objective: None,
            }))
    }
    pub(in crate::builtins::math::optim::optimmodel::lowering) fn into_solution(
        self,
    ) -> RootedValue {
        self.solution
    }
    pub(super) fn from_point(
        point: numerical::Point,
        formulation: &formulation::Formulation,
        policy: Policy,
    ) -> BuiltinResult<Self> {
        let objective = match &formulation.objective {
            Some(objective) => {
                let value = point.at(objective.root, 0, policy)?;
                value.is_finite().then_some(value)
            }
            None => None,
        };
        Ok(Self {
            solution: point.solution,
            objective,
        })
    }
    pub(super) async fn evaluate(
        program: &Program,
        formulation: &formulation::Formulation,
        candidate: &[f64],
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        if let Some(point) =
            numerical::Point::evaluate(program, &formulation.roots, candidate, policy, c).await?
        {
            return Self::from_point(point, formulation, policy).map(Some);
        }
        // Coordinates remain authentic when an original mathematical value is
        // unavailable. Do not fabricate zero F or evaluate the DAG again.
        Ok(numerical::solution(&program.layout, candidate, policy, c)
            .await?
            .map(|solution| Self {
                solution,
                objective: None,
            }))
    }
    pub(in crate::builtins::math::optim::optimmodel::lowering) fn matches<'a>(
        &'a self,
        layout: &'a layout::Layout,
        candidate: &'a [f64],
        policy: Policy,
        c: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<bool>> + 'a {
        numerical::coordinates_match(layout, &self.solution, candidate, policy, c)
    }
}
