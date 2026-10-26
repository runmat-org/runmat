//! A best observation retains its exact point and score provenance. It never
//! replays the objective merely to manufacture a reporting value.
use super::super::callbacks::Invocation;
use super::super::{
    assessment::Assessment,
    evaluation::{
        fitness::Supplied,
        generation::{Evaluated, Violation},
    },
    invalid,
    population::Generation,
    ScalarCooperation, POLICY,
};
use crate::{execution::RootedValue, BuiltinResult};
use std::ops::ControlFlow;

pub(super) struct Best {
    pub point: Option<RootedValue>,
    pub fitness: Option<Supplied>,
    pub violation: f64,
    nonlinear_violation: f64,
    linear_violation: f64,
    pub generation: usize,
    pub inequalities: Vec<f64>,
    pub equalities: Vec<f64>,
}
impl Best {
    pub fn new() -> Self {
        Self {
            point: None,
            fitness: None,
            violation: f64::INFINITY,
            nonlinear_violation: f64::INFINITY,
            linear_violation: f64::INFINITY,
            generation: 0,
            inequalities: Vec::new(),
            equalities: Vec::new(),
        }
    }
    pub fn score(&self) -> Option<f64> {
        self.fitness.and_then(|fitness| fitness.value.score())
    }
    pub fn feasible(&self, tolerance: f64) -> bool {
        self.point.is_some()
            && self.nonlinear_violation <= tolerance
            && self.linear_violation <= super::super::numeric::linear_tolerance(tolerance)
    }
    pub async fn observe<I: Invocation>(
        &mut self,
        population: &Generation,
        evaluated: &Evaluated,
        assessment: &Assessment,
        generation: usize,
        invocation: &mut I,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<ControlFlow<I::Stop, bool>> {
        let Some(row) = assessment.best_feasible else {
            return Ok(ControlFlow::Continue(false));
        };
        let supplied = evaluated.fitness.supplied(row);
        let score = supplied
            .value
            .score()
            .ok_or_else(|| invalid("best feasible row has no finite real fitness"))?;
        if self.score().is_some_and(|old| old <= score) {
            return Ok(ControlFlow::Continue(false));
        }
        let point = match population.point_with(row, invocation, cooperation).await? {
            ControlFlow::Continue(point) => point,
            ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
        };
        let (nonlinear_violation, linear_violation) = match evaluated.violations[row] {
            Violation::Finite {
                maximum, linear, ..
            } => (maximum, linear),
            Violation::Undefined => {
                return Err(invalid("best feasible row has unknown constraints"))
            }
        };
        let (inequality_count, equality_count) = evaluated.constraints.extents();
        let mut inequalities = POLICY.capacity(inequality_count, cooperation.runtime())?;
        let mut equalities = POLICY.capacity(equality_count, cooperation.runtime())?;
        for column in 0..inequality_count {
            inequalities.push(evaluated.constraints.inequality(row, column));
            cooperation.completed(1).await?;
        }
        for column in 0..equality_count {
            equalities.push(evaluated.constraints.equality(row, column));
            cooperation.completed(1).await?;
        }
        self.point = Some(point);
        self.fitness = Some(supplied);
        self.violation = nonlinear_violation.max(linear_violation);
        self.nonlinear_violation = nonlinear_violation;
        self.linear_violation = linear_violation;
        self.generation = generation;
        self.inequalities = inequalities;
        self.equalities = equalities;
        Ok(ControlFlow::Continue(true))
    }
}
