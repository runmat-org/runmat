//! Public checkpoint start/restore joins the same numerical continuation and
//! selected callback owners; no public save/load call or objective replay.
use super::super::{
    controller::{Controller, Setup},
    driver,
    initial::Initial,
    inputs::Raw,
    observers::SelectedObservers,
    options::Options,
    preparation::{self, Plan},
    publication,
};
use super::{error, options::SavedOptions, record, session, ArchivedCallbacks, Session};
use crate::{
    builtins::{
        math::optim::{
            common::{solver_timing::SolverTiming, ScalarCooperation},
            intlinprog::feasibility::Domain,
        },
        stats::random::rng::{self, PortableSnapshot},
    },
    execution::{CallbackBatch, RootedValue, RootedValueSequence},
    BuiltinResult,
};
use runmat_value::Value;
use std::path::PathBuf;

pub(in super::super) async fn start(
    path: String,
    raw: Raw,
    options: Options,
    c: &mut ScalarCooperation,
) -> BuiltinResult<RootedValueSequence> {
    let mut ready = match preparation::prepare_start(raw, options, c).await? {
        Plan::Ordinary(ready) => ready.into_inner(),
        Plan::Inconsistent(variables) => return publication::inconsistent(variables, c).await,
        Plan::Checkpoint(_) => {
            return Err(error(
                "checkpoint preparation did not select a numerical continuation",
            ))
        }
    };
    let _ = c
        .runtime()
        .ok_or_else(SolverTiming::unavailable)?
        .service_ports()
        .timing()?;
    let saved = SavedOptions::capture(&ready.options)?;
    // Validate portable capture and destination staging before the first
    // RNG mutation, objective/observer, UI allocation or pool startup.
    ArchivedCallbacks::capture(
        &ready.selected,
        ready.observers.outputs(),
        ready.observers.plots(),
        c,
    )?;
    let path = session::path(PathBuf::from(path), c)?;
    let reserved = session::reserve(&path, c).await?;
    let clock = SolverTiming::start(c.runtime())?;
    let workers =
        CallbackBatch::select_retained(c.runtime(), &ready.selected, ready.parameters.parallel)?;
    let mut observers = ready.observers.admit_layout(c)?;
    let initial_rng = preparation::begin(ready.rng.take(), c).await?;
    let portable_rng = {
        let runtime = c
            .runtime()
            .ok_or_else(|| error("checkpoint lacks original RNG authority"))?;
        let _entry = runtime.enter();
        PortableSnapshot::capture(&rng::restore_snapshot_value(
            initial_rng.value(),
            Some(runtime),
        )?)
    };
    let mut session = Session::new(path, reserved, saved, ready.initial.clone(), portable_rng);
    let domain = Domain::new(ready.model, clock.clone(), ready.parameters.time, 1e-6, c).await?;
    let mut controller = Controller::new(
        Setup {
            domain,
            source: &ready.source,
            selected: &ready.selected,
            workers: workers.as_ref(),
            clock,
            parameters: ready.parameters,
            linear_tolerance: 1e-6,
            initial: ready.initial,
        },
        c,
    )
    .await?;
    let stop = driver::run_checkpoint(
        &mut controller,
        &mut observers,
        ready.options.display,
        &mut session,
        &ready.selected,
        false,
        c,
    )
    .await?;
    publication::finalize(&controller, stop, &initial_rng, c).await
}

pub(in super::super) async fn resume(
    path_value: Value,
    overrides: Value,
    c: &mut ScalarCooperation,
) -> BuiltinResult<RootedValueSequence> {
    let _ = c
        .runtime()
        .ok_or_else(SolverTiming::unavailable)?
        .service_ports()
        .timing()?;
    let path_text = crate::builtins::common::validation::value_text_scalar_string(&path_value)
        .filter(|path| !path.is_empty())
        .ok_or_else(|| error("checkpoint path must be nonempty scalar text"))?;
    let read_path = session::path(PathBuf::from(path_text), c)?;
    let bytes = {
        let (runtime, budget) = c.runtime_budget();
        match (runtime, budget) {
            (Some(runtime), Some(budget)) => read_path.read(runtime, budget).await?,
            _ => {
                return Err(error(
                    "checkpoint reading lacks original cooperation authority",
                ))
            }
        }
    };
    let runtime = c
        .runtime()
        .cloned()
        .ok_or_else(|| error("checkpoint restoration requires its current original runtime"))?;
    let environment = runtime
        .program_revision()
        .ok_or_else(|| error("checkpoint restoration lacks its current program environment"))?
        .environment();
    let saved = {
        let (runtime, budget) = c.runtime_budget();
        let (Some(runtime), Some(budget)) = (runtime, budget) else {
            return Err(error(
                "checkpoint decoding lacks original cooperation authority",
            ));
        };
        let mut cooperation = super::json::Cooperation::new(runtime, budget)?;
        record::Saved::decode(bytes, &environment, &mut cooperation)
            .await
            .map_err(super::codec_error)?
    };
    if saved.version != record::VERSION || saved.environment != environment {
        return Err(error(
            "checkpoint version or execution environment is incompatible",
        ));
    }
    let overlay = saved.options.overlay(overrides, c).await?;
    let model = saved.geometry.into_input()?;
    let variables = model.lower.len();
    let parameters = overlay.options.controls.resolve(variables)?;
    let initial = match saved.initial {
        Some(initial) => Some(
            initial
                .prepare(variables, parameters.nonlinear_tolerance, c)
                .await?,
        ),
        None => None,
    };
    if let Some(supplied) = overlay.initial_supplied {
        let supplied =
            Initial::prepare(supplied, variables, parameters.nonlinear_tolerance, c).await?;
        match &initial {
            Some(initial) if initial.same_template(&supplied, c).await? => {}
            None if supplied.done() => {}
            _ => {
                return Err(error(
                    "InitialPoints cannot change when resuming a checkpoint",
                ))
            }
        }
    }
    let trials = saved
        .trials
        .decode(variables, parameters.nonlinear_tolerance, c)
        .await?;
    let current_rng = saved.current_rng.restore(&runtime)?;
    let initial_snapshot = saved.initial_rng.restore(&runtime)?;
    let initial_rng_record = PortableSnapshot::capture(&initial_snapshot);
    let initial_rng = {
        let _entry = runtime.enter();
        RootedValue::in_context(rng::snapshot_to_value(initial_snapshot)?, &runtime)?
    };
    let clock = SolverTiming::start(Some(&runtime))?.with_prior_elapsed(saved.frame.elapsed);
    clock.checked_elapsed()?;
    let domain = Domain::new(model, clock.clone(), parameters.time, 1e-6, c).await?;
    let callbacks = saved.callbacks.restore(c)?;
    let mut selected_observers =
        SelectedObservers::from_restored(callbacks.outputs, callbacks.plots);
    if overlay.output_changed {
        selected_observers = selected_observers.with_output(overlay.options.output.as_ref(), c)?;
    }
    if overlay.plot_changed {
        selected_observers = selected_observers.with_plot(overlay.options.plot.as_ref(), c)?;
    }
    ArchivedCallbacks::capture(
        &callbacks.objective,
        selected_observers.outputs(),
        selected_observers.plots(),
        c,
    )?;
    let destination = match &overlay.options.checkpoint {
        Some(path) => session::path(PathBuf::from(path), c)?,
        None => read_path,
    };
    let reserved = if overlay.options.checkpoint.is_some() {
        Some(session::reserve(&destination, c).await?)
    } else {
        None
    };
    let workers =
        CallbackBatch::select_retained(c.runtime(), &callbacks.objective, parameters.parallel)?;
    let mut observers = selected_observers.admit_layout(c)?;
    let mut controller = Controller::new(
        Setup {
            domain,
            source: &callbacks.source,
            selected: &callbacks.objective,
            workers: workers.as_ref(),
            clock,
            parameters,
            linear_tolerance: 1e-6,
            initial: initial.clone(),
        },
        c,
    )
    .await?;
    controller
        .restore_measured(trials, saved.campaign, saved.frame, c)
        .await?;
    validate_notices(
        &saved.notices,
        saved.frame,
        controller.campaign.history(),
        c,
    )
    .await?;
    // All saved numerical/callback/file service admission precedes mutation
    // of the current original RNG stream or any observer/objective execution.
    controller.clock.checked_elapsed()?;
    let _current_rng_value = preparation::begin(Some(current_rng), c).await?;
    let mut session = Session::restored(
        destination,
        reserved,
        overlay.saved,
        initial,
        initial_rng_record,
        saved.notices,
        saved.frame.elapsed,
    );
    let stop = driver::run_checkpoint(
        &mut controller,
        &mut observers,
        overlay.options.display,
        &mut session,
        &callbacks.objective,
        true,
        c,
    )
    .await?;
    publication::finalize(&controller, stop, &initial_rng, c).await
}

async fn validate_notices(
    notices: &[super::super::notice::Frame],
    final_frame: super::super::notice::Frame,
    history: &super::super::history::History,
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let mut prior = (0usize, 0usize, 0usize, std::time::Duration::ZERO, 0u64);
    let mut rows = 0usize;
    let mut actual = 0usize;
    let mut summary = super::super::incumbent::Validation::default();
    for frame in notices.iter().chain(std::iter::once(&final_frame)) {
        frame.validate_shape(history)?;
        let committed = frame.tracker.current.map_or(0, |index| index.0 + 1);
        summary
            .advance(frame.tracker, history, committed, frame.resets, c)
            .await?;
        if frame.rows < prior.0
            || frame.evaluations.0 < prior.1
            || frame.updates.0 < prior.2
            || frame.elapsed < prior.3
            || frame.resets < prior.4
        {
            return Err(error("checkpoint observer chronology is inconsistent"));
        }
        while rows < frame.rows {
            if matches!(
                history
                    .trial(super::super::history::TrialIndex(rows))
                    .source,
                super::super::history::MeasurementSource::Callback
            ) {
                actual = actual
                    .checked_add(1)
                    .ok_or_else(|| error("saved point count overflow"))?;
            }
            rows += 1;
            c.completed(1).await?;
        }
        if actual != frame.evaluations.0 {
            return Err(error(
                "saved observer point count differs from measured provenance",
            ));
        }
        prior = (
            frame.rows,
            frame.evaluations.0,
            frame.updates.0,
            frame.elapsed,
            frame.resets,
        );
    }
    Ok(())
}
