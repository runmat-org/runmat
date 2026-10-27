//! Original-generation adaptive swarm loop; no surrogate objective or local
//! solver stands in for the population algorithm.
use super::{
    common, creation,
    dynamics::Dynamics,
    evaluation::{Counts, Evaluator},
    hybrid, inputs, invalid,
    limits::Timing,
    observation::Observers,
    options::Options,
    random,
    state::{State, Termination},
    ScalarCooperation,
};
use crate::{execution::RootedValue, BuiltinResult};
pub(super) struct Outcome {
    pub state: Option<State>,
    pub point: Option<RootedValue>,
    pub fitness: Option<f64>,
    pub reason: Termination,
    pub counts: Counts,
    pub rng: Option<RootedValue>,
    pub hybrid_flag: Option<f64>,
}
pub(super) async fn minimize(
    mut problem: inputs::Problem,
    mut options: Options,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Outcome> {
    cooperation.completed(0).await?;
    // Pure/domain admission precedes RNG, objective, custom creation or UI.
    let selected_hybrid = hybrid::select(
        options.hybrid.take(),
        &problem.lower,
        &problem.upper,
        cooperation,
    )?;
    if problem.inconsistent {
        return Ok(Outcome {
            state: None,
            point: None,
            fitness: None,
            reason: Termination::InconsistentBounds,
            counts: Counts::default(),
            rng: None,
            hybrid_flag: None,
        });
    }
    let objective = {
        let _scope = cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        crate::canonicalize_callback_handle_for_semantic_resolution(problem.objective.clone())?
    };
    let objective = RootedValue::in_optional_context(objective, cooperation.runtime())?;
    problem.objective = objective.value().clone();
    let creation_root = if let creation::Source::Custom(function) = &options.creation {
        let function = {
            let _scope = cooperation
                .runtime()
                .map(crate::context::RuntimeContext::enter);
            crate::canonicalize_callback_handle_for_semantic_resolution(function.clone())?
        };
        Some(RootedValue::in_optional_context(
            function,
            cooperation.runtime(),
        )?)
    } else {
        None
    };
    if let Some(root) = &creation_root {
        options.creation = creation::Source::Custom(root.value().clone());
    }
    let plot_count = Observers::admit(&options)?;
    if let Some(runtime) = cooperation.runtime() {
        runtime
            .service_ports()
            .require_selected_local_effect(runmat_types::EffectKind::Randomness)?;
        if plot_count > 0 {
            runtime
                .service_ports()
                .require_selected_local_effect(runmat_types::EffectKind::UserInterface)?;
        }
    }
    cooperation.completed(0).await?;
    let mut timing = Timing::new(options.maximum, options.stall_time, cooperation)?;
    let parallel = crate::execution::CallbackBatch::select(
        cooperation.runtime(),
        objective.value(),
        options.parallel,
    )?;
    let observers = Observers::new(&options, cooperation)?;
    let creation_options = options.creation_options(cooperation).await?;
    let public_problem =
        inputs::problem_value(&problem, creation_options.value().clone(), cooperation).await?;
    let rng = random::begin(problem.rng.take(), cooperation).await?;
    let created = creation::create(
        creation::Request {
            source: &options.creation,
            problem: public_problem.value(),
            initial: options.initial.as_ref(),
            members: options.members,
            variables: problem.variables,
            span: &options.span,
            lower: &problem.lower,
            upper: &problem.upper,
        },
        cooperation,
    )
    .await?;
    if created.clipped > 0 && options.display == runmat_builtins::OptimDisplay::Iter {
        let _scope = cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        crate::console::record_console_line(crate::console::ConsoleStream::Stderr,
            format!("Warning: particleswarm clipped {} initial particles to supplied bounds; changed-point Fval entries were not reused.",created.clipped))?;
    }
    let mut evaluator = Evaluator {
        function: objective.value(),
        cooperation,
        vectorized: options.vectorized,
        check_values: options.check_values,
        parallel,
        counts: Counts::default(),
    };
    let fitness = evaluator
        .evaluate(&created.population, created.supplied)
        .await?;
    let scores = fitness.scores(evaluator.cooperation).await?;
    let dynamics = Dynamics::new(
        created.velocity,
        &created.population,
        scores,
        &options.parameters,
    )?;
    let mut state =
        State::new(created.population, fitness, dynamics, evaluator.cooperation).await?;
    timing.improved();
    let mut reason = if observers
        .notify(
            common::output_callback::OutputState::Init,
            &state,
            evaluator.counts,
            &options,
            None,
            evaluator.cooperation,
        )
        .await?
    {
        Some(Termination::Observer)
    } else {
        None
    };
    loop {
        if reason.is_some() {
            break;
        }
        if state.best_fitness < options.objective_limit {
            reason = Some(Termination::Objective);
            break;
        }
        if let Some(stop) = timing.stopped() {
            reason = Some(stop);
            break;
        }
        if state.function_small(options.stall_iterations, options.tolerance) {
            reason = Some(Termination::Function);
            break;
        }
        if state.iteration >= options.iterations {
            reason = Some(Termination::Iterations);
            break;
        }
        state
            .dynamics
            .propose(
                &mut state.population,
                &options.parameters,
                &problem.lower,
                &problem.upper,
                evaluator.cooperation,
            )
            .await?;
        let unsupplied = state.unsupplied(evaluator.cooperation).await?;
        let fitness = evaluator.evaluate(&state.population, unsupplied).await?;
        if state
            .accept(
                fitness,
                &options.parameters,
                options.stall_iterations,
                evaluator.cooperation,
            )
            .await?
        {
            timing.improved();
        }
        if observers
            .notify(
                common::output_callback::OutputState::Iter,
                &state,
                evaluator.counts,
                &options,
                None,
                evaluator.cooperation,
            )
            .await?
        {
            reason = Some(Termination::Observer);
        }
    }
    let reason = reason.ok_or_else(|| invalid("swarm terminated without a reason"))?;
    // The done generation describes the actual final swarm. A later hybrid
    // may improve the returned incumbent but never replaces points.X/Fval.
    observers
        .notify(
            common::output_callback::OutputState::Done,
            &state,
            evaluator.counts,
            &options,
            Some(reason),
            evaluator.cooperation,
        )
        .await?;
    let point = state.best_point(evaluator.cooperation).await?;
    let mut point = RootedValue::in_optional_context(point, evaluator.cooperation.runtime())?;
    let mut fitness = state.best_fitness;
    let mut hybrid_flag = None;
    if reason == Termination::Function {
        if let Some(hybrid) = selected_hybrid {
            let local = hybrid
                .run(
                    objective.value(),
                    point.value().clone(),
                    &problem.lower,
                    &problem.upper,
                    evaluator.cooperation,
                )
                .await?;
            evaluator.counts.points = evaluator
                .counts
                .points
                .checked_add(local.evaluations)
                .ok_or_else(|| invalid("hybrid function count overflow"))?;
            hybrid_flag = Some(local.flag);
            if local.fitness < fitness {
                point = local.point;
                fitness = local.fitness;
            }
        }
    }
    Ok(Outcome {
        state: Some(state),
        point: Some(point),
        fitness: Some(fitness),
        reason,
        counts: evaluator.counts,
        rng: Some(rng),
        hybrid_flag,
    })
}
