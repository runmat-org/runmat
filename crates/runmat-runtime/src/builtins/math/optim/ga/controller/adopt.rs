//! A returned callback state is a declaration for its population, not an
//! objective replay or permission to replace the original constraint model.
use super::super::common::callback_source::Callable;
use super::super::common::population::projection::Arguments;
use super::super::linear::Geometry;
use super::super::{
    assessment::Assessment,
    common::population::Fitness,
    evaluation::{
        duplicates::Aliases,
        fitness::{Objective, Origin, Supplied},
        generation::{self, Evaluator},
    },
    invalid,
    options::{Options, Prepared},
    population::Generation,
    reproduction, ScalarCooperation, POLICY,
};
use super::{
    best::Best,
    score_value::{self, Role},
    search::{self, Search},
    state_input::{self, Mode, State},
};
use crate::{execution::RootedValue, BuiltinResult};
use runmat_builtins::{OptimGeneticConstraint, OptimGeneticStateField as Field};

pub(super) struct Adoption {
    pub changed: bool,
    pub population: bool,
}

impl<G: Geometry> Search<G> {
    #[cfg(test)]
    pub async fn adopt(
        &mut self,
        state: &RootedValue,
        controls: Option<&RootedValue>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Adoption> {
        self.adopt_with(
            state,
            controls,
            &mut super::super::callbacks::Direct,
            cooperation,
        )
        .await
        .map(super::super::callbacks::take_direct)
    }
    pub async fn adopt_with<S: super::super::evaluation::source::Source>(
        &mut self,
        state_root: &RootedValue,
        controls: Option<&RootedValue>,
        source: &mut S,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<std::ops::ControlFlow<S::Stop, Adoption>> {
        cooperation.completed(0).await?;
        let state = State::new(state_root)?;
        let mode = mode(self);
        // No returned numeric role transfers before all consumed state roles
        // have passed their compatibility and authenticated residency gates.
        state.preflight(self.options.population, mode, None, cooperation)?;
        let replacement = if let Some(controls) = controls {
            let prepared = Prepared::continuation(
                controls.value().clone(),
                super::super::options::Activity::from_problem(&self.problem),
                cooperation,
            )
            .await?;
            if prepared.population != self.options.population {
                return Err(invalid(
                    "changing PopulationType during an active search requires a new ga invocation",
                ));
            }
            let bounds = if let Some(domain) = &self.domain {
                domain.bounds()
            } else {
                let data = self
                    .problem
                    .linear
                    .as_ref()
                    .ok_or_else(|| invalid("original population bounds are unavailable"))?;
                (&data.lower[..], &data.upper[..])
            };
            let mut options =
                Options::with_bounds(prepared, &self.problem, bounds, cooperation).await?;
            if options.constraint != self.options.constraint {
                return Err(invalid("changing NonlinearConstraintAlgorithm during an active search requires a new ga invocation"));
            }
            options.bind_generation_callbacks(cooperation)?;
            Some(options)
        } else {
            None
        };
        let options = replacement.as_ref().unwrap_or(&self.options);
        let hybrid = replacement
            .as_ref()
            .map(|options| {
                super::super::hybrid::select(
                    options.field(super::super::options::Field::HybridFcn),
                    &self.problem,
                    options.population,
                    cooperation,
                )
            })
            .transpose()?;
        let members = options.layout.total();
        state.preflight(options.population, mode, Some(members), cooperation)?;
        let flags = state.flags(&self.flags, cooperation).await?;
        let changed_population = members != self.generation.members()
            || !state_input::same_numeric(
                state.require(Field::Population)?,
                &self.generation.value(),
            );
        let previous_scores = if matches!(mode, Mode::Raw) {
            self.evaluated.fitness.values.value()
        } else {
            self.assessment.public_scores.clone()
        };
        let changed_score =
            !state_input::same_numeric(state.require(Field::Score)?, &previous_scores);
        let changed_fitness = matches!(mode, Mode::Penalty)
            && !state_input::same_numeric(
                state.require(Field::Fitness)?,
                &self.evaluated.fitness.values.value(),
            );
        if !changed_population && !changed_score && !changed_fitness && replacement.is_none() {
            self.flags = flags;
            return Ok(std::ops::ControlFlow::Continue(Adoption {
                changed: false,
                population: false,
            }));
        }
        let parallel = if options.parallel != self.options.parallel
            || options.vectorized != self.options.vectorized
        {
            let routes = source.select_routes(
                &self.problem.objective,
                self.problem.nonlinear.as_ref(),
                options.vectorized,
                options.parallel,
                cooperation,
            )?;
            Some((routes.objective, routes.nonlinear))
        } else {
            None
        };
        let (objective_parallel, nonlinear_parallel) =
            if let Some((objective, nonlinear)) = &parallel {
                (objective.as_ref(), nonlinear.as_ref())
            } else {
                (
                    self.objective_parallel.as_ref(),
                    self.nonlinear_parallel.as_ref(),
                )
            };
        let population = if changed_population {
            Some(
                match Generation::from_result_with(
                    state.root(Field::Population, cooperation)?,
                    members,
                    self.problem.variables,
                    options.population,
                    source,
                    cooperation,
                )
                .await?
                {
                    std::ops::ControlFlow::Continue(population) => population,
                    std::ops::ControlFlow::Break(stop) => {
                        return Ok(std::ops::ControlFlow::Break(stop))
                    }
                },
            )
        } else {
            None
        };
        let generation = population.as_ref().unwrap_or(&self.generation);
        let tolerance_changed = options.constraint_tolerance != self.options.constraint_tolerance;
        let score = if changed_score || changed_population && !matches!(mode, Mode::Augmented) {
            Some(
                score_value::decode(
                    state.require(Field::Score)?.clone(),
                    Role::Score,
                    members,
                    options.check_values,
                    cooperation,
                )
                .await?,
            )
        } else {
            None
        };
        let (raw, score) = match mode {
            Mode::Raw => (score, None),
            Mode::Penalty if changed_fitness || changed_population => (
                Some(
                    score_value::decode(
                        state.require(Field::Fitness)?.clone(),
                        Role::Fitness,
                        members,
                        options.check_values,
                        cooperation,
                    )
                    .await?,
                ),
                score,
            ),
            _ => (None, score),
        };
        if !changed_population && tolerance_changed {
            // A tightened tolerance cannot silently authorize an existing
            // linear-infeasible state as a new admissible generation.
            super::linear::admit(
                generation,
                self.domain.as_mut(),
                options.constraint_tolerance,
                cooperation,
            )
            .await?;
        }
        let evaluated = if changed_population {
            let maxima = super::linear::admit(
                generation,
                self.domain.as_mut(),
                options.constraint_tolerance,
                cooperation,
            )
            .await?;
            let aliases = Aliases::new(generation, flags.duplicates, cooperation).await?;
            let supplied = supplied(raw.as_ref(), members, cooperation).await?;
            let evaluated = Evaluator {
                objective: Callable::Selected(&self.sources.objective),
                nonlinear: self.sources.nonlinear.as_ref().map(Callable::Selected),
                objective_parallel,
                nonlinear_parallel,
                vectorized: options.vectorized,
                check_values: options.check_values,
                mixed_integer: self.problem.mixed_integer,
                mode: search::evaluation_mode(&self.problem, options),
                tolerance: options.constraint_tolerance,
                objective_counts: &mut self.objective_counts,
                nonlinear_counts: &mut self.nonlinear_counts,
                extents: &mut self.extents,
                aliases: aliases.as_ref(),
                cooperation,
            }
            .evaluate_with(generation, supplied, source)
            .await?;
            let mut evaluated = match evaluated {
                std::ops::ControlFlow::Continue(evaluated) => evaluated,
                std::ops::ControlFlow::Break(stop) => {
                    return Ok(std::ops::ControlFlow::Break(stop))
                }
            };
            super::linear::combine(&mut evaluated, maxima, cooperation).await?;
            Some(evaluated)
        } else {
            None
        };
        // A declared score is retained with its own provenance. It is not
        // charged as an objective call or relabelled a measured observation.
        let mut previous_evaluated = None;
        let mut previous_fitness = None;
        if let Some(evaluated) = evaluated {
            previous_evaluated = Some(std::mem::replace(&mut self.evaluated, evaluated));
        } else if raw.is_some() || tolerance_changed {
            let mut values = if raw.is_some() {
                supplied(raw.as_ref(), members, cooperation).await?
            } else {
                let mut values = POLICY.capacity(members, cooperation.runtime())?;
                for row in 0..members {
                    values.push(Some(self.evaluated.fitness.supplied(row)));
                    cooperation.completed(1).await?;
                }
                values
            };
            generation::admit_supplied(
                &self.evaluated.violations,
                search::evaluation_mode(&self.problem, options),
                options.constraint_tolerance,
                &mut values,
                cooperation,
            )
            .await?;
            // The immutable physical constraint observations still describe
            // this exact population; only newly required objectives run.
            let fitness = Objective {
                function: Callable::Selected(&self.sources.objective),
                vectorized: options.vectorized,
                check_values: options.check_values,
                parallel: objective_parallel,
                counts: &mut self.objective_counts,
                aliases: None,
                cooperation,
            }
            .evaluate_with(generation, values, source)
            .await?;
            let fitness = match fitness {
                std::ops::ControlFlow::Continue(fitness) => fitness,
                std::ops::ControlFlow::Break(stop) => {
                    return Ok(std::ops::ControlFlow::Break(stop))
                }
            };
            previous_fitness = Some(std::mem::replace(&mut self.evaluated.fitness, fitness));
        }
        let replacement_merit = if self.merit.is_some()
            && (options.penalty != self.options.penalty
                || options.penalty_factor != self.options.penalty_factor)
        {
            Some(search::initial_merit(&self.evaluated, options, cooperation).await?)
        } else {
            None
        };
        let mut assessment = Assessment::new(
            &self.evaluated,
            search::assessment_mode(
                &self.problem,
                options,
                replacement_merit.as_ref().or(self.merit.as_ref()),
            ),
            options.constraint_tolerance,
            cooperation,
        )
        .await?;
        if !matches!(mode, Mode::Raw) {
            if let Some(score) = score {
                assessment.override_scores(score, cooperation).await?;
            }
        }
        let previous_best = if !self.best.feasible(options.constraint_tolerance) {
            Some(std::mem::replace(&mut self.best, Best::new()))
        } else {
            None
        };
        let observed = self
            .best
            .observe(
                generation,
                &self.evaluated,
                &assessment,
                self.progress.generation,
                source,
                cooperation,
            )
            .await?;
        if let std::ops::ControlFlow::Break(stop) = observed {
            if let Some(previous) = previous_best {
                self.best = previous;
            }
            if let Some(previous) = previous_evaluated {
                self.evaluated = previous;
            } else if let Some(previous) = previous_fitness {
                self.evaluated.fitness = previous;
            }
            return Ok(std::ops::ControlFlow::Break(stop));
        }
        if let Some(merit) = replacement_merit {
            self.merit = Some(merit);
            self.progress.reset_merit(options);
        }
        self.progress.invalidate_comparisons(options);
        if let Some(population) = population {
            self.generation = population;
        }
        if let Some(domain) = &mut self.domain {
            domain.retune(options.maximum, options.constraint_tolerance)?;
        }
        self.assessment = assessment;
        self.flags = flags;
        if let Some((objective, nonlinear)) = parallel {
            self.objective_parallel = objective;
            self.nonlinear_parallel = nonlinear;
        }
        if let Some(options) = replacement {
            self.hybrid = hybrid.expect("replacement options select their own hybrid generation");
            self.mutation = reproduction::Mutation::new(&options, cooperation).await?;
            self.options = options;
        }
        source.retune(self.options.maximum);
        Ok(std::ops::ControlFlow::Continue(Adoption {
            changed: true,
            population: changed_population,
        }))
    }
}
fn mode<G>(search: &Search<G>) -> Mode {
    if search.merit.is_some() {
        Mode::Augmented
    } else if search.problem.mixed_integer
        || search.problem.nonlinear.is_some()
            && search.options.constraint == OptimGeneticConstraint::Penalty
    {
        Mode::Penalty
    } else {
        Mode::Raw
    }
}
async fn supplied(
    fitness: Option<&Fitness>,
    members: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<Option<Supplied>>> {
    let mut values = POLICY.capacity(members, cooperation.runtime())?;
    for row in 0..members {
        values.push(fitness.map(|fitness| Supplied {
            value: fitness.measurement(row),
            origin: Origin::ObserverSupplied,
        }));
        cooperation.completed(1).await?;
    }
    Ok(values)
}
