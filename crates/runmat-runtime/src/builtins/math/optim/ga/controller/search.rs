//! One numerical GA lifecycle; callbacks and publication use this same owner.
use super::super::evaluation::source::Source;
use super::super::{
    assessment::{self, Assessment},
    common, creation,
    evaluation::{
        duplicates::Aliases,
        generation::{Evaluated, Evaluator, Mode},
    },
    inputs::Problem,
    lineage::Lineage,
    merit::AugmentedLagrangian,
    options::Options,
    outcome::Termination,
    population::Generation,
    reproduction, ScalarCooperation,
};
use super::{best::Best, frame::Flags, linear, progress::Progress, supplied};
use crate::{execution::CallbackBatch, BuiltinResult};
use common::callback_source::Callable;
use common::{population::Counts, solver_timing::SolverTiming};
use runmat_builtins::{OptimGeneticAction, OptimGeneticConstraint, OptimGeneticPopulation};
use runmat_value::{Tensor, Value};
use std::ops::ControlFlow;
mod start;
mod retained;
pub(super) use retained::Retained;
#[cfg(test)]
mod tests;

pub(super) struct Search<G = super::super::linear::Domain> {
    pub problem: Problem,
    pub options: Options,
    pub timing: SolverTiming,
    pub generation: Generation,
    pub domain: Option<G>,
    pub evaluated: Evaluated,
    pub assessment: Assessment,
    pub merit: Option<AugmentedLagrangian>,
    pub objective_counts: Counts,
    pub nonlinear_counts: Counts,
    pub extents: Option<(usize, usize)>,
    pub best: Best,
    pub progress: Progress,
    pub flags: Flags,
    pub supplied: usize,
    pub sources: super::super::callbacks::Sources,
    pub hybrid: Option<super::super::hybrid::Selected>,
    pub action: Option<OptimGeneticAction>,
    pub(super) objective_parallel: Option<CallbackBatch>,
    pub(super) nonlinear_parallel: Option<CallbackBatch>,
    pub(super) mutation: reproduction::Mutation,
    pub(super) merit_row: Option<super::nonlinear::Row>,
}
pub(super) struct InitialStop {
    pub reason: Termination,
    pub nonlinear_calls: usize,
    pub nonlinear_points: usize,
}
pub(super) struct Start<G = super::super::linear::Domain> {
    pub ready: Option<Retained<G>>,
    pub stopped: Option<InitialStop>,
}
pub(super) struct Record {
    pub expectation: Option<Tensor>,
    pub selection: Vec<usize>,
    pub lineage: Vec<Lineage>,
    pub ancestors: Vec<Option<super::super::lineage::Parents>>,
}
pub(super) enum Step {
    Completed(Record),
    Stopped(Termination),
}

fn augmented(problem: &Problem, options: &Options) -> bool {
    problem.nonlinear.is_some() && options.constraint == OptimGeneticConstraint::AugmentedLagrangian
}
pub(super) fn evaluation_mode(problem: &Problem, options: &Options) -> Mode {
    if problem.mixed_integer
        || problem.nonlinear.is_some() && options.constraint == OptimGeneticConstraint::Penalty
    {
        Mode::Penalty
    } else {
        Mode::Raw
    }
}
pub(super) fn assessment_mode<'a>(
    problem: &Problem,
    options: &Options,
    merit: Option<&'a AugmentedLagrangian>,
) -> assessment::Mode<'a> {
    if let Some(merit) = merit {
        assessment::Mode::Augmented(merit)
    } else if matches!(evaluation_mode(problem, options), Mode::Penalty) {
        assessment::Mode::Penalty
    } else {
        assessment::Mode::Raw
    }
}
impl<G: super::super::linear::Geometry> Search<G> {
    pub async fn step_with<S: Source>(
        &mut self,
        state: Option<&Value>,
        controls: Option<&Value>,
        record: bool,
        source: &mut S,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<ControlFlow<S::Stop, Step>> {
        let runtime = super::continuation::original(cooperation)?;
        super::continuation::run(&runtime, move || {
            self.step_phase(state, controls, record, source, cooperation)
        })?
        .await
    }
    async fn step_phase<S: Source>(
        &mut self,
        state: Option<&Value>,
        controls: Option<&Value>,
        record: bool,
        source: &mut S,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<ControlFlow<S::Stop, Step>> {
        cooperation.completed(0).await?;
        if let Some(reason) = self.progress.hard_stop(&self.options, &self.timing) {
            return Ok(ControlFlow::Continue(Step::Stopped(reason)));
        }
        let (generation, generations) = self.progress.next_schedule(&self.options)?;
        if !self
            .mutation
            .begin(generation, generations, cooperation)
            .await?
        {
            return Ok(ControlFlow::Continue(Step::Stopped(
                Termination::Unverified,
            )));
        }
        let mut flags = Flags {
            eval_elites: self.flags.eval_elites,
            duplicates: self.flags.duplicates,
        };
        let successor = reproduction::next_with(
            reproduction::Current {
                generation: &self.generation,
                comparison: &self.assessment.comparison,
                public_scores: &self.assessment.public_scores,
                controls,
                state,
                problem: &self.problem.public,
                options: &self.options,
                timing: &self.timing,
                record,
            },
            self.domain.as_mut(),
            &mut self.mutation,
            source,
            cooperation,
        )
        .await?;
        let successor = match successor {
            ControlFlow::Continue(value) => value,
            ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
        };
        let Some(mut successor) = successor else {
            return Ok(ControlFlow::Continue(Step::Stopped(Termination::Time)));
        };
        if successor.unverified_proposals {
            return Ok(ControlFlow::Continue(Step::Stopped(
                Termination::Unverified,
            )));
        }
        let supplied = supplied::copied(
            &successor.copied_from,
            &self.evaluated.fitness,
            &self.options,
            &flags,
            cooperation,
        )
        .await?;
        let mut evaluated = {
            let maxima = linear::admit(
                &successor.generation,
                self.domain.as_mut(),
                self.options.constraint_tolerance,
                cooperation,
            )
            .await?;
            let aliases =
                Aliases::new(&successor.generation, flags.duplicates, cooperation).await?;
            if let Some(aliases) = &aliases {
                flags.duplicates = aliases.selected().len() < aliases.positions().len();
            }
            let evaluated = Evaluator {
                objective: Callable::Selected(&self.sources.objective),
                nonlinear: self.sources.nonlinear.as_ref().map(Callable::Selected),
                objective_parallel: self.objective_parallel.as_ref(),
                nonlinear_parallel: self.nonlinear_parallel.as_ref(),
                vectorized: self.options.vectorized,
                check_values: self.options.check_values,
                mixed_integer: self.problem.mixed_integer,
                mode: evaluation_mode(&self.problem, &self.options),
                tolerance: self.options.constraint_tolerance,
                objective_counts: &mut self.objective_counts,
                nonlinear_counts: &mut self.nonlinear_counts,
                extents: &mut self.extents,
                aliases: aliases.as_ref(),
                cooperation,
            }
            .evaluate_with(&successor.generation, supplied, source)
            .await?;
            let mut evaluated = match evaluated {
                ControlFlow::Continue(value) => value,
                ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
            };
            linear::combine(&mut evaluated, maxima, cooperation).await?;
            evaluated
        };
        if flags.eval_elites
            && supplied::repeated_elites_match(
                &successor.copied_from,
                &self.evaluated.fitness,
                &evaluated.fitness,
                &self.options,
                cooperation,
            )
            .await?
        {
            flags.eval_elites = false;
        }
        let mut assessment = Assessment::new(
            &evaluated,
            assessment_mode(&self.problem, &self.options, self.merit.as_ref()),
            self.options.constraint_tolerance,
            cooperation,
        )
        .await?;
        let observed = self
            .best
            .observe(
                &successor.generation,
                &evaluated,
                &assessment,
                self.progress.following_generation()?,
                source,
                cooperation,
            )
            .await?;
        if let ControlFlow::Break(stop) = observed {
            return Ok(ControlFlow::Break(stop));
        }
        let comparison = assessment.best_comparison();
        let mut record = Record {
            expectation: successor.expectation,
            selection: successor.selection,
            lineage: successor.lineage,
            ancestors: successor.ancestors,
        };
        if let ControlFlow::Break(stop) = self
            .migrate(
                &mut successor.generation,
                &mut evaluated,
                &mut assessment,
                &mut record,
                source,
                cooperation,
            )
            .await?
        {
            return Ok(ControlFlow::Break(stop));
        }
        let improved = self
            .progress
            .observed_comparison(comparison, true, true, &self.timing, cooperation)
            .await?;
        // Publish only the complete successor, including any semantic migration.
        self.generation = successor.generation;
        self.evaluated = evaluated;
        self.assessment = assessment;
        self.mutation.completed(improved);
        self.flags = flags;
        Ok(ControlFlow::Continue(Step::Completed(record)))
    }
}

pub(super) async fn initial_merit(
    evaluated: &Evaluated,
    options: &Options,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<AugmentedLagrangian> {
    let (inequalities, equalities) = evaluated.constraints.extents();
    let mut maxima = super::super::POLICY.capacity(inequalities, cooperation.runtime())?;
    for column in 0..inequalities {
        let mut maximum: Option<f64> = None;
        for row in 0..evaluated.constraints.rows {
            let value = evaluated.constraints.inequality(row, column);
            if value.is_finite() {
                maximum = Some(maximum.map_or(value, |old| old.max(value)));
            }
            cooperation.completed(1).await?;
        }
        maxima.push(maximum);
    }
    AugmentedLagrangian::new(
        &maxima,
        equalities,
        options.penalty,
        options.penalty_factor,
        cooperation,
    )
    .await
}
