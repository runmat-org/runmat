//! Completed-generation stopping and fixed-merit AL epochs. Public outer
//! generations and inner interrupt iterations are separate real counters.
use super::super::{
    assessment::Assessment, common::solver_timing::SolverTiming, invalid, options::Options,
    outcome::Termination, stall::Window, ScalarCooperation,
};
use crate::BuiltinResult;
use std::time::Duration;

/// RunMat's bounded inexact subproblem policy uses the ordinary GA default.
/// It is not an option, a resource grant or a MATLAB trajectory equivalence.
const INNER_GENERATIONS_PER_VARIABLE: usize = 100;
pub(super) enum Subproblem {
    Continue,
    Completed { solved: bool },
}
pub(super) struct Progress {
    pub generation: usize,
    pub inner: usize,
    pub total_inner: usize,
    pub last_improvement: usize,
    pub last_improvement_time: Duration,
    augmented: bool,
    inner_budget: usize,
    best_comparison: Option<f64>,
    window: Window,
}
impl Progress {
    pub fn new(options: &Options, variables: usize, augmented: bool) -> BuiltinResult<Self> {
        let inner_budget = if augmented {
            variables
                .checked_mul(INNER_GENERATIONS_PER_VARIABLE)
                .ok_or_else(|| invalid("nonlinear subproblem generation budget overflow"))?
        } else {
            0
        };
        Ok(Self {
            generation: 0,
            inner: 0,
            total_inner: 0,
            last_improvement: 0,
            last_improvement_time: Duration::ZERO,
            augmented,
            inner_budget,
            best_comparison: None,
            window: Window::new(options.stall_generations),
        })
    }
    pub fn following_generation(&self) -> BuiltinResult<usize> {
        if self.augmented {
            Ok(self.generation)
        } else {
            self.generation
                .checked_add(1)
                .ok_or_else(|| invalid("generation count overflow"))
        }
    }
    pub fn following_migration_generation(&self) -> BuiltinResult<usize> {
        if self.augmented {
            self.total_inner
                .checked_add(1)
                .ok_or_else(|| invalid("total inner generation count overflow"))?;
            self.inner
                .checked_add(1)
                .ok_or_else(|| invalid("inner generation count overflow"))
        } else {
            self.following_generation()
        }
    }
    pub fn observed<'a>(
        &'a mut self,
        assessment: &Assessment,
        completed_generation: bool,
        new_observation: bool,
        timing: &'a SolverTiming,
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<bool>> + 'a {
        self.observed_comparison(
            assessment.best_comparison(),
            completed_generation,
            new_observation,
            timing,
            cooperation,
        )
    }
    pub async fn observed_comparison(
        &mut self,
        comparison: Option<f64>,
        completed_generation: bool,
        new_observation: bool,
        timing: &SolverTiming,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if completed_generation {
            if self.augmented {
                self.inner = self
                    .inner
                    .checked_add(1)
                    .ok_or_else(|| invalid("inner generation count overflow"))?;
                self.total_inner = self
                    .total_inner
                    .checked_add(1)
                    .ok_or_else(|| invalid("total inner generation count overflow"))?;
            } else {
                self.generation = self.following_generation()?;
            }
        }
        let improved = new_observation
            && comparison.is_some_and(|value| self.best_comparison.is_none_or(|old| value < old));
        if improved {
            self.best_comparison = comparison;
            self.last_improvement = self.generation;
            self.last_improvement_time = timing.elapsed();
        } else if !new_observation {
            // A new merit's baseline is not an observed fitness improvement.
            self.best_comparison = comparison;
        }
        self.window.observe(comparison, cooperation).await?;
        Ok(improved)
    }
    pub fn hard_stop(&self, options: &Options, timing: &SolverTiming) -> Option<Termination> {
        let elapsed = timing.elapsed();
        if options.maximum.reached(elapsed) {
            Some(Termination::Time)
        } else if options
            .stall_time
            .reached(elapsed.saturating_sub(self.last_improvement_time))
        {
            Some(Termination::StallTime)
        } else {
            None
        }
    }
    pub fn ordinary_stop(
        &self,
        options: &Options,
        feasible_fitness: Option<f64>,
    ) -> Option<Termination> {
        if feasible_fitness.is_some_and(|value| value <= options.fitness_limit) {
            return Some(Termination::FitnessLimit);
        }
        // MaxGenerations=0 is initial-only even when AL's zero-sized stall
        // window would otherwise complete a vacuous inner subproblem.
        if self.generation >= options.generations {
            return Some(Termination::Generations);
        }
        if !self.augmented
            && self
                .window
                .small(options.stall_test, options.function_tolerance)
        {
            return Some(if feasible_fitness.is_none() {
                Termination::NoFeasiblePoint
            } else if options.stall_generations == 0
                || options.function_tolerance == 0. && self.window.unchanged()
            {
                Termination::Unchanged
            } else {
                Termination::Function
            });
        }
        None
    }
    pub fn subproblem(&self, options: &Options) -> Subproblem {
        if !self.augmented {
            return Subproblem::Continue;
        }
        if self
            .window
            .small(options.stall_test, options.function_tolerance)
        {
            Subproblem::Completed {
                solved: self.inner != 0,
            }
        } else if self.inner >= self.inner_budget {
            Subproblem::Completed { solved: false }
        } else {
            Subproblem::Continue
        }
    }
    pub fn complete_subproblem(&mut self) -> BuiltinResult<()> {
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| invalid("outer generation count overflow"))?;
        Ok(())
    }
    pub fn reset_merit(&mut self, options: &Options) {
        self.inner = 0;
        self.best_comparison = None;
        self.window.reset(options.stall_generations);
    }
    pub fn invalidate_comparisons(&mut self, options: &Options) {
        // User-supplied selection values and changed comparison policies do
        // not establish an objective improvement or a solved AL subproblem.
        self.best_comparison = None;
        self.window.reset(options.stall_generations);
    }
    pub fn next_schedule(&self, options: &Options) -> BuiltinResult<(usize, usize)> {
        let (current, limit) = if self.augmented {
            (self.inner, self.inner_budget)
        } else {
            (self.generation, options.generations)
        };
        let next = current
            .checked_add(1)
            .ok_or_else(|| invalid("mutation schedule generation overflow"))?;
        if next > limit {
            return Err(invalid("no mutation generation remains in this subproblem"));
        }
        Ok((next, limit))
    }
}
