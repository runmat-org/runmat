//! Accepted-generation lifecycle, bounded real probes and honest final state.
use super::{
    common,
    evaluation::Evaluator,
    hybrid,
    inputs::{Domain, Problem},
    invalid,
    observation::{Notice, Observers},
    operators,
    options::{HybridInterval, Options},
    random, reanneal,
    state::{Stall, State, Termination},
    ScalarCooperation,
};
use crate::{execution::RootedValue, BuiltinResult};
use common::output_callback::OutputState;
pub(super) struct Outcome {
    pub generation: Generation,
    pub rng: RootedValue,
    pub count: usize,
    pub reason: Termination,
    pub elapsed: f64,
    pub bounded: bool,
}
pub(super) enum Generation {
    Measured(Box<State>),
    Unmeasured {
        point: RootedValue,
        temperature: Vec<f64>,
    },
    Infeasible,
}
pub(super) async fn minimize(
    problem: Problem,
    mut options: Options,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Outcome> {
    let custom = options.datatype == runmat_builtins::OptimAnnealData::Custom;
    let (domain, inconsistent) = Domain::admit(&problem, custom, c).await?;
    options.resolve(domain.variables(), c).await?;
    hybrid::admit(&options, &domain, c)?;
    let timing = common::solver_timing::SolverTiming::start(c.runtime())?;
    let mut observers = Observers::new(&options, c)?;
    let rng = random::begin(problem.rng, c).await?;
    let bounded = domain.bounded();
    if inconsistent {
        return Ok(Outcome {
            generation: Generation::Infeasible,
            rng,
            count: 0,
            reason: Termination::Infeasible,
            elapsed: timing.elapsed().as_secs_f64(),
            bounded,
        });
    }
    let current = domain.initial(problem.initial.value(), c).await?;
    let mut evaluator = Evaluator {
        function: problem.objective,
        count: 0,
    };
    // The single immutable problem metadata generation is shared by custom
    // proposals. It retains the original x0, not a mutable current-point copy.
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("objective", evaluator.function.value().clone());
    fields.insert("x0", problem.initial.value().clone());
    fields.insert("nvar", runmat_value::Value::Num(domain.variables() as f64));
    fields.insert("lb", problem.lower);
    fields.insert("ub", problem.upper);
    let callback_problem = RootedValue::in_optional_context(
        runmat_value::Value::Struct(fields.finish()),
        c.runtime(),
    )?;
    if options.evaluations == 0 {
        return Ok(Outcome {
            generation: Generation::Unmeasured {
                point: current,
                temperature: options.initial_temperature,
            },
            rng,
            count: 0,
            reason: Termination::Budget,
            elapsed: timing.elapsed().as_secs_f64(),
            bounded,
        });
    }
    let current_f = evaluator.evaluate(current.value(), c).await?;
    let mut state = State {
        best: RootedValue::in_optional_context(current.value().clone(), c.runtime())?,
        current,
        current_f,
        best_f: current_f,
        temperature: options.initial_temperature.clone(),
        k: vec![0.0; domain.variables()],
        iteration: 0,
        accepted: 0,
        stall: Stall::new(options.stall),
    };
    let mut reason = if observers
        .notify(
            Notice {
                phase: OutputState::Init,
                count: evaluator.count,
                started: timing.started().as_secs_f64(),
                reason: None,
            },
            &state,
            &mut options,
            c,
        )
        .await?
    {
        Some(Termination::Observer)
    } else {
        None
    };
    let mut stall_width = options.stall;
    hybrid::admit(&options, &domain, c)?;
    while reason.is_none() {
        if state.best_f < options.objective {
            reason = Some(Termination::Objective);
            break;
        }
        if options
            .iterations
            .is_some_and(|limit| state.iteration >= limit)
            || evaluator.count >= options.evaluations
        {
            reason = Some(Termination::Budget);
            break;
        }
        if options.maximum.reached(timing.elapsed()) {
            reason = Some(Termination::Time);
            break;
        }
        let previous = state.current_f;
        let trial = operators::proposal(
            &state,
            &domain,
            &callback_problem,
            &options,
            timing.started().as_secs_f64(),
            evaluator.count,
            c,
        )
        .await?;
        let fitness = evaluator.evaluate(trial.value(), c).await?;
        let accepted = operators::accept(
            &state,
            &trial,
            fitness,
            &options,
            timing.started().as_secs_f64(),
            evaluator.count,
            c,
        )
        .await?;
        // Every measured proposal can improve the best, even when a custom
        // acceptance function rejects it as the next Markov-chain point.
        if fitness < state.best_f {
            state.best = RootedValue::in_optional_context(trial.value().clone(), c.runtime())?;
            state.best_f = fitness;
        }
        if accepted {
            state.current = trial;
            state.current_f = fitness;
            state.accepted = state
                .accepted
                .checked_add(1)
                .ok_or_else(|| invalid("accepted count overflow"))?;
        }
        state.iteration = state
            .iteration
            .checked_add(1)
            .ok_or_else(|| invalid("iteration count overflow"))?;
        if state.best_f < options.objective {
            reason = Some(Termination::Objective);
        }
        if let HybridInterval::Iterations(interval) = options.hybrid_interval {
            if reason.is_none() && state.iteration.is_multiple_of(interval) {
                hybrid::run(&mut state, &domain, &options, &mut evaluator, c).await?;
            }
        }
        let full = state.stall.push(previous, state.current_f, c).await?;
        if reason.is_none() && full && state.stall.average() < options.tolerance {
            reason = Some(Termination::Stall);
        }
        if reason.is_none()
            && options.reanneal > 0
            && state.accepted >= options.reanneal
            && !reanneal::refresh(&mut state, &domain, &options, &mut evaluator, c).await?
        {
            reason = Some(Termination::Budget);
        }
        operators::cool(
            &mut state,
            &options,
            timing.started().as_secs_f64(),
            evaluator.count,
            c,
        )
        .await?;
        if options.maximum.reached(timing.elapsed()) {
            reason = Some(Termination::Time);
        }
        let stop = observers
            .notify(
                Notice {
                    phase: OutputState::Iter,
                    count: evaluator.count,
                    started: timing.started().as_secs_f64(),
                    reason,
                },
                &state,
                &mut options,
                c,
            )
            .await?;
        if stop {
            reason = Some(Termination::Observer);
        }
        if options.stall != stall_width {
            state.stall.reset(options.stall);
            stall_width = options.stall;
        }
        hybrid::admit(&options, &domain, c)?;
        c.completed(1).await?;
    }
    let mut reason = reason.expect("termination");
    if matches!(options.hybrid_interval, HybridInterval::End)
        && matches!(
            reason,
            Termination::Stall | Termination::Objective | Termination::Budget
        )
    {
        hybrid::run(&mut state, &domain, &options, &mut evaluator, c).await?;
        if state.best_f < options.objective {
            reason = Termination::Objective;
        }
        if options.maximum.reached(timing.elapsed()) {
            reason = Termination::Time;
        }
    }
    observers
        .notify(
            Notice {
                phase: OutputState::Done,
                count: evaluator.count,
                started: timing.started().as_secs_f64(),
                reason: Some(reason),
            },
            &state,
            &mut options,
            c,
        )
        .await?;
    Ok(Outcome {
        generation: Generation::Measured(Box::new(state)),
        rng,
        count: evaluator.count,
        reason,
        elapsed: timing.elapsed().as_secs_f64(),
        bounded,
    })
}
