//! Initial validation and evaluation retire before the controller is polled.
use super::*;
pub(super) async fn prepare(
    input: &Input<'_>,
    prepared: data::Prepared,
    options: Options,
    domain: observation::Domain,
    policy: numeric::Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Preparation> {
    let data::Prepared {
        initial,
        shape,
        scalar,
        linear,
        inconsistent_bounds,
        short_bounds,
        search_window,
        original_bounds,
        ..
    } = prepared;
    if matches!(
        options.controls.search,
        Some(super::super::search::Search::Nelder { .. })
    ) && (linear.matrix.m > 0 || input.constraints.is_some())
    {
        return Err(invalid(
            "searchneldermead supports unconstrained problems only",
        ));
    }
    let mutable_time = common::output_callback::callback_count(
        options.callback(Field::OutputFcn),
        common::output_callback::CallbackForm::Handles,
    )? > 0
        && options.controls.evaluations > 0;
    let genetic_time = matches!(
        options.controls.search,
        Some(super::super::search::Search::Genetic { .. })
    ) && options.controls.iterations > 0
        && options.controls.evaluations > 0;
    let deadline = Deadline::new(
        options.controls.maximum_time,
        cooperation.runtime(),
        mutable_time || genetic_time,
    )?;
    if inconsistent_bounds {
        return Ok(Preparation::Finished((
            Outcome {
                engine: None,
                initial: None,
                reason: Stop::Infeasible,
                objective_calls: 0,
                constraint_calls: 0,
                violation: None,
            },
            None,
            options,
        )));
    }
    let rng = if options.controls.polling.method().is_none()
        || input.rngstate.is_some()
        || mutable_time
        || options.controls.search.is_some_and(|search| {
            matches!(
                search,
                super::super::search::Search::Latin { .. }
                    | super::super::search::Search::Genetic { .. }
            )
        })
        || crate::output_count::current_output_count_in(cooperation.runtime())
            .is_some_and(|n| n >= 4)
    {
        Some(random::begin(input.rngstate, cooperation).await?)
    } else {
        None
    };
    if short_bounds {
        let _scope = cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        crate::warnings::emit(crate::warnings::WarningRequest {
            builtin: super::super::NAME,
            identifier: "RunMat:patternsearch:ShortBounds",
            message: "Shorter bounds apply to their supplied leading coordinates; remaining coordinates are unbounded.",
            show_identifier: true,
        })?;
    }
    let initial = match phase_one::feasible_start(
        initial,
        &linear,
        options.controls.constraint_tolerance,
        IDENTITY,
        || deadline.reached(),
        cooperation,
    )
    .await?
    {
        Ok(initial) => initial,
        Err(failure) => {
            let reason = match failure.reason {
                phase_one::Failure::Infeasible => Stop::Infeasible,
                phase_one::Failure::Numerical => Stop::Numerical,
                phase_one::Failure::TimeLimit => Stop::Time,
            };
            return Ok(Preparation::Finished((
                Outcome {
                    engine: None,
                    initial: if reason == Stop::Infeasible {
                        None
                    } else {
                        Some(failure.initial)
                    },
                    reason,
                    objective_calls: 0,
                    constraint_calls: 0,
                    violation: None,
                },
                rng,
                options,
            )));
        }
    };
    // A zero point limit does not invoke either callback. An initial point is
    // still a meaningful output, but its unmeasured objective is never invented.
    if options.controls.evaluations == 0 {
        let violation = if input.constraints.is_none() {
            linear.violation(&initial, cooperation).await?
        } else {
            None
        };
        return Ok(Preparation::Finished((
            Outcome {
                engine: None,
                initial: Some(initial),
                reason: Stop::Limit,
                objective_calls: 0,
                constraint_calls: 0,
                violation,
            },
            rng,
            options,
        )));
    }
    let execution = evaluation::batch::Execution::select(
        &options.controls,
        input.objective,
        input.constraints,
        cooperation.runtime(),
    )?;
    let mut evaluation_state = evaluation::State {
        callbacks: evaluation::CallbackCounts::default(),
        limit: options.controls.evaluations,
        objective_calls: 0,
        constraint_calls: 0,
        extents: None,
        history: if options.controls.search == Some(super::super::search::Search::Rbf) {
            Some(super::super::search::rbf::History::new(initial.len(), 0)?)
        } else {
            None
        },
        original_bounds,
        execution,
        cache: options
            .controls
            .cache
            .map(super::super::cache::Cache::new)
            .transpose()?,
    };
    let mut evaluator = evaluation::Evaluator {
        context: evaluation::Context {
            execution: None,
            demand: evaluation::ObservationDemand::MaySkipVisited,
            objective: input.objective,
            constraints: input.constraints,
            shape: &shape,
            scalar,
            policy,
        },
        state: &mut evaluation_state,
        cooperation,
    };
    let observers = observation::Observers::new(&options, &evaluator)?;
    let point = match evaluator.evaluate(initial).await? {
        Ok(point) => point,
        Err(evaluation::Stop::Undefined) => return Err(invalid(
            "the objective and nonlinear constraints must be finite and real at the initial point",
        )),
        Err(evaluation::Stop::Budget) => {
            return Err(invalid(
                "initial callback budget admission changed unexpectedly",
            ))
        }
        Err(evaluation::Stop::Cached) => return Err(invalid("initial cache must be empty")),
    };
    let engine = iteration::Engine::new(
        point,
        &options.controls,
        input.constraints.is_some(),
        search_window,
        &mut evaluator,
    )
    .await?;

    Ok(Preparation::Ready(Ready {
        engine,
        options,
        linear,
        domain,
        policy,
        shape,
        scalar,
        state: evaluation_state,
        deadline,
        observers,
        rng,
    }))
}
