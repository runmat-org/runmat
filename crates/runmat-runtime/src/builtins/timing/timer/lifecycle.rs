//! Session-owned asynchronous timer lifecycle.

use super::*;
use crate::context::LocalTaskCommandOrigin;
use crate::context::RuntimeLocalTaskOutcome;
use crate::context::{TimerAdmissionError, TimerFailureReceipt, TimerPhase};
use runmat_async::RuntimeErrorOwner;
use std::time::Duration;

mod schedule;
use schedule::{run_timer_callbacks, still_running};

#[derive(Clone, Copy)]
struct TimerRunConfig {
    busy_mode: TimerBusyMode,
    execution_mode: TimerExecutionMode,
    visibility: TimerVisibility,
    start_delay: Duration,
    period: Duration,
    tasks: usize,
    requires_source_evaluation: bool,
}

impl TimerRunConfig {
    fn admit(
        runtime: &RuntimeContext,
        handle: &HandleRef,
        override_delay: Option<Duration>,
    ) -> BuiltinResult<Self> {
        ensure_valid_timer(handle)?;
        if callback_is_empty(&property(handle, TimerCallback::Timer.property().name())?) {
            return Err(timer_error(
                &TIMER_ERROR_INVALID_PROPERTY,
                "timer: TimerFcn must be set before starting a timer",
            ));
        }
        let mut requires_source_evaluation = false;
        for slot in TimerCallback::ALL {
            let callback = property(handle, slot.property().name())?;
            if let Some(text) = nonempty_text_callback(&callback)? {
                // Capture through the executor's source catalog, then release
                // the preflight lease. Each actual firing captures its own
                // source identity so callback property edits remain visible.
                let request = text_callback_request(runtime, handle, slot, &text)?;
                drop(request);
                requires_source_evaluation = true;
            }
        }
        let busy_mode =
            TimerBusyMode::parse(&string_property(handle, "BusyMode")?).ok_or_else(|| {
                timer_error(
                    &TIMER_ERROR_INVALID_PROPERTY,
                    "timer: invalid BusyMode value",
                )
            })?;
        let execution_mode = TimerExecutionMode::parse(&string_property(handle, "ExecutionMode")?)
            .ok_or_else(|| {
                timer_error(
                    &TIMER_ERROR_INVALID_PROPERTY,
                    "timer: invalid ExecutionMode value",
                )
            })?;
        let visibility = TimerVisibility::parse(&string_property(handle, "ObjectVisibility")?)
            .ok_or_else(|| {
                timer_error(
                    &TIMER_ERROR_INVALID_PROPERTY,
                    "timer: invalid ObjectVisibility value",
                )
            })?;
        let period = numeric_property(handle, "Period")?;
        let start_delay = match override_delay {
            Some(delay) => delay,
            None => duration_property(numeric_property(handle, "StartDelay")?, "StartDelay")?,
        };
        let period = duration_property(period, "Period")?;
        let tasks = if execution_mode == TimerExecutionMode::SingleShot {
            1
        } else {
            parse_tasks_to_execute_value(&property(handle, "TasksToExecute")?)?
        };
        Ok(Self {
            busy_mode,
            execution_mode,
            visibility,
            start_delay,
            period,
            tasks,
            requires_source_evaluation,
        })
    }
}

pub(super) fn duration_property(seconds: f64, name: &str) -> BuiltinResult<Duration> {
    Duration::try_from_secs_f64(seconds).map_err(|_| {
        timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            format!("timer: {name} is outside the supported duration range"),
        )
    })
}

fn nonempty_text_callback(value: &Value) -> BuiltinResult<Option<std::borrow::Cow<'_, str>>> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(text)) if !text.trim().is_empty() => {
            Some(std::borrow::Cow::Borrowed(text))
        }
        Value::CharArray(chars) if chars.data().iter().any(|ch| !ch.is_whitespace()) => {
            Some(std::borrow::Cow::Owned(
                crate::builtins::common::text::decode_utf16_with_error(chars.data(), || {
                    timer_error(
                        &TIMER_ERROR_INVALID_PROPERTY,
                        "timer: callback text is not valid UTF-16",
                    )
                })?,
            ))
        }
        _ => None,
    })
}

fn admission_error(error: TimerAdmissionError) -> RuntimeError {
    match error {
        TimerAdmissionError::ForeignOrDeleted => timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            "timer: handle belongs to another session or was deleted",
        ),
        TimerAdmissionError::AlreadyRunning => timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            "timer: timer is already running",
        ),
        TimerAdmissionError::GenerationExhausted => timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            "timer: lifecycle generation exhausted",
        ),
    }
}

pub(super) async fn start_one_timer(
    runtime: &RuntimeContext,
    handle: HandleRef,
    override_delay: Option<Duration>,
) -> BuiltinResult<()> {
    if runtime.timer_session().borrow().record(&handle).is_none() {
        return Err(admission_error(TimerAdmissionError::ForeignOrDeleted));
    }
    let config = TimerRunConfig::admit(runtime, &handle, override_delay)?;
    runtime
        .service_ports()
        .require_local_tasks("timer")
        .map_err(crate::context::RuntimeCapabilityError::into_runtime_error)?;
    let task_runtime = background_callback_runtime(runtime, config.requires_source_evaluation)?;
    // The session record owns discovery, while this independent root keeps a
    // currently polled callback's receiver alive if it deletes its timer.
    let task_root = runmat_gc::gc_root(handle.target)
        .map_err(|err| timer_error(&TIMER_ERROR_GC, format!("timer: {err}")))?;
    let (generation, failure) = runtime
        .timer_session()
        .borrow_mut()
        .admit_start(
            &handle,
            config.busy_mode,
            config.execution_mode,
            config.visibility,
        )
        .map_err(admission_error)?;
    let presentation = set_properties(
        runtime,
        &handle,
        &[
            ("TasksExecuted", Value::Num(0.0)),
            ("InstantPeriod", Value::Num(f64::NAN)),
            ("AveragePeriod", Value::Num(f64::NAN)),
            ("Running", Value::String("on".into())),
        ],
    );
    if let Err(error) = presentation {
        runtime.timer_session().borrow_mut().stop(&handle);
        let _ = set_property(runtime, &handle, "Running", Value::String("off".into()));
        return Err(error);
    }
    let task_handle = handle.clone();
    let lease = match runtime.spawn_local_with_command_origin(
        "timer",
        LocalTaskCommandOrigin::CurrentSelected,
        move |selected| {
            let task_runtime =
                task_runtime.with_cleanup_command_origin(selected.command_profiling().cloned());
            let scoped_runtime = task_runtime.clone();
            Box::pin(scoped_runtime.scope(async move {
                let _task_root = task_root;
                run_timer_task(task_runtime, task_handle, generation, config, failure).await;
            }))
        },
    ) {
        Ok(lease) => lease,
        Err(error) => {
            runtime.timer_session().borrow_mut().stop(&handle);
            let _ = set_property(runtime, &handle, "Running", Value::String("off".into()));
            return Err(error);
        }
    };
    if let Err(lease) = runtime
        .timer_session()
        .borrow_mut()
        .attach_lease(&handle, generation, lease)
    {
        lease.cancel();
        return Err(timer_error(
            &TIMER_ERROR_INVALID_HANDLE,
            "timer: start was cancelled before its task was admitted",
        ));
    }
    Ok(())
}

fn background_callback_runtime(
    runtime: &RuntimeContext,
    requires_source_evaluation: bool,
) -> BuiltinResult<RuntimeContext> {
    runtime.background_callback_runtime(requires_source_evaluation, "timer text callback")
}

async fn run_timer_task(
    runtime: RuntimeContext,
    handle: HandleRef,
    generation: u64,
    config: TimerRunConfig,
    failure: TimerFailureReceipt,
) {
    let outcome = run_timer_callbacks(&runtime, &handle, generation, config).await;
    let terminal_error = outcome.err().map(RuntimeErrorOwner::new);
    if let Some(error) = &terminal_error {
        failure.record(error.clone());
        if still_running(&runtime, &handle, generation) {
            let _ = run_error_callback(&runtime, &handle, error.clone()).await;
        }
        tracing::warn!(target: "runmat_timer", "timer callback failed: {}", error);
    }
    let finish = runtime
        .timer_session()
        .borrow_mut()
        .finish(&handle, generation);
    if let Some(call_stop) = finish {
        let _ = set_property(&runtime, &handle, "Running", Value::String("off".into()));
        if call_stop {
            let _ = run_callback(&runtime, &handle, TimerCallback::Stop).await;
        }
    }
}

pub(super) async fn stop_one_timer(
    runtime: &RuntimeContext,
    handle: &HandleRef,
    call_stop: bool,
) -> BuiltinResult<()> {
    ensure_valid_timer(handle)?;
    let (lease, claimed) = runtime
        .timer_session()
        .borrow_mut()
        .stop(handle)
        .ok_or_else(|| admission_error(TimerAdmissionError::ForeignOrDeleted))?;
    if let Some(lease) = lease {
        lease.cancel();
    }
    set_property(runtime, handle, "Running", Value::String("off".into()))?;
    if call_stop && claimed {
        run_callback(runtime, handle, TimerCallback::Stop).await?;
    }
    Ok(())
}

pub(super) async fn wait_one_timer(
    runtime: &RuntimeContext,
    handle: &HandleRef,
) -> BuiltinResult<()> {
    ensure_valid_timer(handle)?;
    loop {
        let (generation, lease, failure) = runtime
            .timer_session()
            .borrow()
            .record(handle)
            .map(|record| {
                (
                    record.generation,
                    record.lease.clone(),
                    record.failure.clone(),
                )
            })
            .ok_or_else(|| admission_error(TimerAdmissionError::ForeignOrDeleted))?;
        let outcome = if let Some(lease) = lease {
            Some(lease.join().await)
        } else {
            None
        };
        if let Some(error) = failure.take() {
            return Err(RuntimeError::from_shared_original(error));
        }
        let current_generation = runtime
            .timer_session()
            .borrow()
            .record(handle)
            .map(|record| record.generation)
            .ok_or_else(|| admission_error(TimerAdmissionError::ForeignOrDeleted))?;
        if current_generation != generation {
            continue;
        }
        if outcome == Some(RuntimeLocalTaskOutcome::Cancelled) {
            let same_generation = runtime
                .timer_session()
                .borrow()
                .record(handle)
                .is_some_and(|record| record.generation == generation);
            if same_generation {
                let _ = runtime.timer_session().borrow_mut().stop(handle);
                let _ = set_property(runtime, handle, "Running", Value::String("off".into()));
            }
            return Err(timer_error(
                &TIMER_ERROR_CALLBACK,
                "timer: callback task was cancelled before completion",
            ));
        }
        return Ok(());
    }
}
