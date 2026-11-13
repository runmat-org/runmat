//! Persistent mutation schedules belong to this solver invocation, not to
//! callbacks or a global operator-name table.
use super::super::{
    common, invalid, linear,
    operators::{feasible, mutation},
    options, random, ScalarCooperation, POLICY,
};
use crate::BuiltinResult;
use common::solver_timing::SolverTiming;
use runmat_builtins::{OptimGeneticMutation, OptimGeneticPopulation};

pub(in super::super) enum Proposal {
    Point(Vec<f64>),
    Retain,
    TimeLimit,
    Unverified,
}
pub(in super::super) struct Mutation {
    gaussian: Option<mutation::Gaussian>,
    scaling: Option<feasible::Scaling>,
    adaptive: feasible::Adaptive,
    intervals: Vec<(f64, f64)>,
}
impl Mutation {
    pub async fn new(
        options: &options::Options,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let kind = options.mutation.kind();
        let gaussian = if kind == Some(OptimGeneticMutation::Gaussian) {
            let extra = options.mutation.extra();
            Some(
                mutation::Gaussian::new(
                    &options.ranges,
                    parameter(extra.first(), 1.)?,
                    parameter(extra.get(1), 1.)?,
                    cooperation,
                )
                .await?,
            )
        } else {
            None
        };
        let scaling = if matches!(
            kind,
            Some(OptimGeneticMutation::AdaptiveFeasible | OptimGeneticMutation::PositiveBasis)
        ) {
            Some(feasible::Scaling::new(&options.ranges, cooperation).await?)
        } else {
            None
        };
        let intervals = POLICY.capacity(
            if kind == Some(OptimGeneticMutation::Power) {
                options.ranges.len()
            } else {
                0
            },
            cooperation.runtime(),
        )?;
        Ok(Self {
            gaussian,
            scaling,
            adaptive: feasible::Adaptive::new(),
            intervals,
        })
    }
    /// Install the specified generation's Gaussian deviation before its first
    /// child. The outer AL count is not the fixed subproblem's schedule.
    pub async fn begin(
        &mut self,
        generation: usize,
        generations: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if let Some(gaussian) = &mut self.gaussian {
            gaussian.advance(generation, generations, cooperation).await
        } else {
            Ok(true)
        }
    }
    pub fn completed(&mut self, improved: bool) {
        self.adaptive.completed_generation(improved);
    }
    pub async fn propose<G: linear::Geometry>(
        &mut self,
        parent: &[f64],
        mut domain: Option<&mut G>,
        options: &options::Options,
        timing: &SolverTiming,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Proposal> {
        let kind = options
            .mutation
            .kind()
            .ok_or_else(|| invalid("custom mutation uses the semantic operator owner"))?;
        if options.maximum.reached(timing.elapsed()) {
            return Ok(Proposal::TimeLimit);
        }
        match kind {
            OptimGeneticMutation::Gaussian => {
                let draws = random::normal(parent.len(), cooperation).await?;
                Ok(
                    match self
                        .gaussian
                        .as_ref()
                        .ok_or_else(|| invalid("Gaussian schedule is unavailable"))?
                        .child(parent, &draws, cooperation)
                        .await?
                    {
                        Some(point) => Proposal::Point(point),
                        None => Proposal::Retain,
                    },
                )
            }
            OptimGeneticMutation::Uniform => {
                let rate = parameter(options.mutation.extra().first(), 0.01)?;
                if options.population == OptimGeneticPopulation::BitString {
                    let draws = random::uniform(parent.len(), cooperation).await?;
                    return mutation::bit_flip(parent, rate, &draws, cooperation)
                        .await
                        .map(Proposal::Point);
                }
                let count = parent
                    .len()
                    .checked_mul(2)
                    .ok_or_else(|| invalid("mutation draw extent overflow"))?;
                let draws = random::uniform(count, cooperation).await?;
                mutation::uniform(parent, &options.ranges, rate, &draws, cooperation)
                    .await
                    .map(Proposal::Point)
            }
            OptimGeneticMutation::Power => {
                let bounds = domain
                    .as_ref()
                    .ok_or_else(|| invalid("integer mutation lacks original geometry"))?
                    .bounds();
                if bounds.0.len() != parent.len() {
                    return Err(invalid("mutation bound extent changed"));
                }
                self.intervals.clear();
                for (column, &value) in parent.iter().enumerate() {
                    let lower = if bounds.0[column].is_finite() {
                        bounds.0[column]
                    } else {
                        options.ranges[column].0.min(value)
                    };
                    let upper = if bounds.1[column].is_finite() {
                        bounds.1[column]
                    } else {
                        options.ranges[column].1.max(value)
                    };
                    self.intervals.push((lower, upper));
                    cooperation.completed(1).await?;
                }
                let count = parent
                    .len()
                    .checked_mul(2)
                    .ok_or_else(|| invalid("mutation draw extent overflow"))?;
                let draws = random::uniform(count, cooperation).await?;
                mutation::power(parent, &self.intervals, &draws, cooperation)
                    .await
                    .map(Proposal::Point)
            }
            OptimGeneticMutation::AdaptiveFeasible | OptimGeneticMutation::PositiveBasis => {
                let domain = domain
                    .as_mut()
                    .ok_or_else(|| invalid("feasible mutation lacks original geometry"))?;
                let scaling = self
                    .scaling
                    .as_ref()
                    .ok_or_else(|| invalid("feasible mutation scaling is unavailable"))?;
                let step = self
                    .adaptive
                    .propose(
                        &mut **domain,
                        feasible::Request {
                            kind,
                            parent,
                            scaling,
                            tolerance: options.constraint_tolerance,
                            timing,
                            deadline: options.maximum,
                        },
                        cooperation,
                    )
                    .await?;
                Ok(match step {
                    linear::step::Step::Point(point) => Proposal::Point(point),
                    linear::step::Step::Blocked => Proposal::Retain,
                    linear::step::Step::TimeLimit => Proposal::TimeLimit,
                    linear::step::Step::Unverified => Proposal::Unverified,
                })
            }
        }
    }
}
fn parameter(value: Option<&runmat_value::Value>, default: f64) -> BuiltinResult<f64> {
    value
        .map(|value| options::real(options::Owner::Solver, value))
        .transpose()
        .map(|value| value.unwrap_or(default))
}
