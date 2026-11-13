//! Constraint transactions precede penalty fitness transactions. These are
//! separate observations, with independent real callback and point counters.
use super::super::{common, invalid, ScalarCooperation, POLICY};
use super::{
    constraints::Constraints,
    fitness::{Objective, Origin, RawFitness, Supplied},
    nonlinear::Nonlinear,
};
use crate::{execution::CallbackBatch, BuiltinResult};
use common::callback_source::Callable;
use common::population::{projection::Arguments, Counts, Measurement};

#[derive(Clone, Copy)]
pub(in super::super) enum Mode {
    Raw,
    Penalty,
}
#[derive(Clone, Copy)]
pub(in super::super) enum Violation {
    Undefined,
    Finite { maximum: f64, sum: f64, linear: f64 },
}
impl Violation {
    pub fn admitted(self, nonlinear_tolerance: f64) -> bool {
        matches!(self, Self::Finite { maximum, linear, .. }
            if maximum <= nonlinear_tolerance
                && linear <= super::super::numeric::linear_tolerance(nonlinear_tolerance))
    }
}
pub(in super::super) struct Evaluated {
    pub fitness: RawFitness,
    pub constraints: Constraints,
    pub violations: Vec<Violation>,
}
pub(in super::super) struct Evaluator<'a> {
    pub objective: Callable<'a>,
    pub nonlinear: Option<Callable<'a>>,
    pub objective_parallel: Option<&'a CallbackBatch>,
    pub nonlinear_parallel: Option<&'a CallbackBatch>,
    pub vectorized: bool,
    pub check_values: bool,
    pub mixed_integer: bool,
    pub mode: Mode,
    pub tolerance: f64,
    pub objective_counts: &'a mut Counts,
    pub nonlinear_counts: &'a mut Counts,
    pub extents: &'a mut Option<(usize, usize)>,
    pub aliases: Option<&'a super::duplicates::Aliases<'a>>,
    pub cooperation: &'a mut ScalarCooperation,
}
use std::ops::ControlFlow;

impl Evaluator<'_> {
    #[cfg(test)]
    pub async fn evaluate(
        &mut self,
        population: &super::super::population::Generation,
        supplied: Vec<Option<Supplied>>,
    ) -> BuiltinResult<Evaluated> {
        self.evaluate_with(population, supplied, &mut super::source::Direct)
            .await
            .map(super::super::callbacks::take_direct)
    }
    pub async fn evaluate_with<S: super::source::Source>(
        &mut self,
        population: &super::super::population::Generation,
        mut supplied: Vec<Option<Supplied>>,
        source: &mut S,
    ) -> BuiltinResult<ControlFlow<S::Stop, Evaluated>> {
        if supplied.len() != population.members()
            || !self.tolerance.is_finite()
            || self.tolerance < 0.
        {
            return Err(invalid(
                "invalid generation fitness/constraint extent or tolerance",
            ));
        }
        let constraints = Nonlinear {
            function: self.nonlinear,
            vectorized: self.vectorized,
            parallel: self.nonlinear_parallel,
            counts: self.nonlinear_counts,
            extents: self.extents,
            mixed_integer: self.mixed_integer,
            cooperation: self.cooperation,
            aliases: self.aliases,
        }
        .evaluate(population)
        .await?;
        let mut violations = POLICY.capacity(population.members(), self.cooperation.runtime())?;
        for row in 0..supplied.len() {
            let violation = constraints.row_violation(row, self.cooperation).await?;
            violations.push(match violation {
                Some((maximum, sum)) => Violation::Finite {
                    maximum,
                    sum,
                    linear: 0.,
                },
                None => Violation::Undefined,
            });
            self.cooperation.completed(1).await?;
        }
        admit_supplied(
            &violations,
            self.mode,
            self.tolerance,
            &mut supplied,
            self.cooperation,
        )
        .await?;
        let fitness = Objective {
            function: self.objective,
            vectorized: self.vectorized,
            check_values: self.check_values,
            parallel: self.objective_parallel,
            counts: self.objective_counts,
            cooperation: self.cooperation,
            aliases: self.aliases,
        }
        .evaluate_with(population, supplied, source)
        .await?;
        let fitness = match fitness {
            ControlFlow::Continue(value) => value,
            ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
        };
        Ok(ControlFlow::Continue(Evaluated {
            fitness,
            constraints,
            violations,
        }))
    }
}
pub(in super::super) async fn admit_supplied(
    violations: &[Violation],
    mode: Mode,
    tolerance: f64,
    supplied: &mut [Option<Supplied>],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    if violations.len() != supplied.len() || !tolerance.is_finite() || tolerance < 0. {
        return Err(invalid("fitness admission extent or tolerance changed"));
    }
    for (&violation, fitness) in violations.iter().zip(supplied) {
        // A prior rejection is not an objective observation. A point newly
        // admitted by a changed tolerance must run its actual objective.
        if fitness.is_some_and(|fitness| fitness.origin == Origin::ConstraintRejected) {
            *fitness = None;
        }
        if matches!(mode, Mode::Penalty) && !violation.admitted(tolerance) {
            *fitness = Some(Supplied {
                value: Measurement::Real(f64::INFINITY),
                origin: Origin::ConstraintRejected,
            });
        }
        cooperation.completed(1).await?;
    }
    Ok(())
}
impl Evaluated {
    pub async fn augmented_scores(
        &self,
        merit: &super::super::merit::AugmentedLagrangian,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<super::super::operators::ranking::Score>> {
        use super::super::{merit::BarrierDomain, operators::ranking::Score};
        let (inequalities, equalities) = self.constraints.extents();
        // One reusable row scratch, not a copied constraint generation.
        let mut c = POLICY.capacity(inequalities, cooperation.runtime())?;
        let mut ceq = POLICY.capacity(equalities, cooperation.runtime())?;
        let mut scores = POLICY.capacity(self.constraints.rows, cooperation.runtime())?;
        for row in 0..self.constraints.rows {
            c.clear();
            ceq.clear();
            for column in 0..inequalities {
                c.push(self.constraints.inequality(row, column));
                cooperation.completed(1).await?;
            }
            for column in 0..equalities {
                ceq.push(self.constraints.equality(row, column));
                cooperation.completed(1).await?;
            }
            scores.push(match merit.domain(&c) {
                BarrierDomain::Undefined => Score::Undefined,
                BarrierDomain::Exterior(violation) => Score::RestoreDomain(violation),
                BarrierDomain::Interior => self
                    .fitness
                    .values
                    .measurement(row)
                    .score()
                    .and_then(|fitness| merit.score(fitness, &c, &ceq))
                    .map_or(Score::Undefined, Score::Defined),
            });
            cooperation.completed(1).await?;
        }
        Ok(scores)
    }
    pub async fn penalty_scores(
        &self,
        tolerance: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<Option<f64>>> {
        if !tolerance.is_finite() || tolerance < 0. {
            return Err(invalid("invalid penalty constraint tolerance"));
        }
        let mut worst: Option<f64> = None;
        for (row, violation) in self.violations.iter().enumerate() {
            if violation.admitted(tolerance) {
                if let Some(value) = self.fitness.values.measurement(row).score() {
                    worst = Some(worst.map_or(value, |old| old.max(value)));
                }
            }
            cooperation.completed(1).await?;
        }
        let mut scores = POLICY.capacity(self.violations.len(), cooperation.runtime())?;
        for (row, violation) in self.violations.iter().enumerate() {
            scores.push(match violation {
                Violation::Undefined => None,
                Violation::Finite { sum, .. } => super::super::merit::penalty_score(
                    self.fitness.values.measurement(row).score(),
                    *sum,
                    violation.admitted(tolerance),
                    worst,
                ),
            });
            cooperation.completed(1).await?;
        }
        Ok(scores)
    }
}
