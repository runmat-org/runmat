//! Session-owned asynchronous timer lifecycle.

use super::*;
use crate::context::RuntimeLocalTaskOutcome;
use crate::context::{TimerAdmissionError, TimerPhase};
use std::time::Duration;

#[derive(Clone, Copy)]
struct TimerRunConfig {
    busy_mode: TimerBusyMode,
    execution_mode: TimerExecutionMode,
    visibility: TimerVisibility,
    start_delay: Duration,
    period: Duration,
    tasks: usize,
}

impl TimerRunConfig {
    fn admit(handle: &HandleRef, override_delay: Option<f64>) -> BuiltinResult<Self> {
        ensure_valid_timer(handle)?;
        if callback_is_empty(&property(handle, "TimerFcn")?) {
            return Err(timer_error(
                &TIMER_ERROR_INVALID_PROPERTY,
                "timer: TimerFcn must be set before starting a timer",
            ));
        }
        for callback_name in CALLBACK_PROPS {
            let callback = property(handle, callback_name)?;
            if nonempty_text_callback(&callback) {
                return Err(timer_error(
                    &TIMER_ERROR_CALLBACK,
                    "timer: text callback execution requires the session source-evaluation service",
                ));
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
        let delay = override_delay.unwrap_or(numeric_property(handle, "StartDelay")?);
        let period = numeric_property(handle, "Period")?;
        let start_delay = duration_property(delay, "StartDelay")?;
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
        })
    }
}

fn duration_property(seconds: f64, name: &str) -> BuiltinResult<Duration> {
    Duration::try_from_secs_f64(seconds).map_err(|_| {
        timer_error(
            &TIMER_ERROR_INVALID_PROPERTY,
            format!("timer: {name} is outside the supported duration range"),
        )
    })
}

fn nonempty_text_callback(value: &Value) -> bool {
    match value {
        Value::String(text) => !text.trim().is_empty(),
        Value::CharArray(chars) => chars.data.iter().any(|ch| !ch.is_whitespace()),
        _ => false,
    }
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
    override_delay: Option<f64>,
) -> BuiltinResult<()> {
    if runtime.timer_session().borrow().record(&handle).is_none() {
        return Err(admission_error(TimerAdmissionError::ForeignOrDeleted));
    }
    let config = TimerRunConfig::admit(&handle, override_delay)?;
    let tasks = runtime.service_ports().require_local_tasks("timer")?;
    let generation = runtime
        .timer_session()
        .borrow_mut()
        .admit_start(
            &handle,
            config.busy_mode,
            config.execution_mode,
            config.visibility,
        )
        .map_err(admission_error)?;
    let presentation = (|| {
        set_property(&handle, "TasksExecuted", Value::Num(0.0))?;
        set_property(&handle, "InstantPeriod", Value::Num(f64::NAN))?;
        set_property(&handle, "AveragePeriod", Value::Num(f64::NAN))?;
        set_property(&handle, "Running", Value::String("on".into()))
    })();
    if let Err(error) = presentation {
        runtime.timer_session().borrow_mut().stop(&handle);
        let _ = set_property(&handle, "Running", Value::String("off".into()));
        return Err(error);
    }
    let task_runtime = runtime.clone();
    let scoped_runtime = runtime.clone();
    let task_handle = handle.clone();
    let future = scoped_runtime.scope(async move {
        run_timer_task(task_runtime, task_handle, generation, config).await;
    });
    let lease = match tasks.spawn_local(runtime.local_identity(), Box::pin(future)) {
        Ok(lease) => lease,
        Err(error) => {
            runtime.timer_session().borrow_mut().stop(&handle);
            let _ = set_property(&handle, "Running", Value::String("off".into()));
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

async fn run_timer_task(
    runtime: RuntimeContext,
    handle: HandleRef,
    generation: u64,
    config: TimerRunConfig,
) {
    let outcome = run_timer_callbacks(&runtime, &handle, generation, config).await;
    if let Err(error) = &outcome {
        let _ = run_callback(&runtime, &handle, "ErrorFcn", "ErrorFcn").await;
        tracing::warn!(target: "runmat_timer", "timer callback failed: {}", error);
    }
    let finish = runtime
        .timer_session()
        .borrow_mut()
        .finish(&handle, generation, outcome.err());
    if let Some(call_stop) = finish {
        let _ = set_property(&handle, "Running", Value::String("off".into()));
        if call_stop {
            let _ = run_callback(&runtime, &handle, "StopFcn", "StopFcn").await;
        }
    }
}

async fn run_timer_callbacks(
    runtime: &RuntimeContext,
    handle: &HandleRef,
    generation: u64,
    config: TimerRunConfig,
) -> BuiltinResult<()> {
    let (execution_mode, busy_mode) = runtime
        .timer_session()
        .borrow()
        .record(handle)
        .filter(|entry| entry.generation == generation)
        .map(|entry| (entry.execution_mode, entry.busy_mode))
        .ok_or_else(|| admission_error(TimerAdmissionError::ForeignOrDeleted))?;
    run_callback(runtime, handle, "StartFcn", "StartFcn").await?;
    if !still_running(runtime, handle, generation) {
        return Ok(());
    }
    sleep_until(runtime, config.start_delay).await?;
    let timing = runtime.service_ports().timing().clone();
    let mut cooperation = runtime.cpu_cooperation();
    let mut scheduled = 0usize;
    let mut executed = 0usize;
    let mut last_fire = None;
    let mut next_deadline = timing.monotonic_elapsed();
    while scheduled < config.tasks {
        if !still_running(runtime, handle, generation) {
            return Ok(());
        }
        let fired = timing.monotonic_elapsed();
        if let Some(last) = last_fire {
            let instant = fired.saturating_sub(last).as_secs_f64();
            set_property(handle, "InstantPeriod", Value::Num(instant))?;
            update_average_period(handle, instant, executed)?;
        }
        last_fire = Some(fired);
        run_callback(runtime, handle, "TimerFcn", "TimerFcn").await?;
        if !still_running(runtime, handle, generation) {
            return Ok(());
        }
        executed += 1;
        scheduled += 1;
        set_property(handle, "TasksExecuted", Value::Num(executed as f64))?;
        cooperation
            .checkpoint(1)
            .await
            .map_err(crate::execution::ExecutionServiceError::into_runtime_error)?;
        if scheduled >= config.tasks || execution_mode == TimerExecutionMode::SingleShot {
            break;
        }
        let now = timing.monotonic_elapsed();
        next_deadline = match execution_mode {
            TimerExecutionMode::SingleShot => Some(now),
            TimerExecutionMode::FixedSpacing => now.checked_add(config.period),
            TimerExecutionMode::FixedDelay => fired.checked_add(config.period),
            TimerExecutionMode::FixedRate => next_deadline.checked_add(config.period),
        }
        .ok_or_else(|| {
            timer_error(
                &TIMER_ERROR_INVALID_PROPERTY,
                "timer: schedule deadline overflow",
            )
        })?;
        if execution_mode == TimerExecutionMode::FixedRate && now > next_deadline {
            match busy_mode {
                TimerBusyMode::Drop => {
                    let period = config.period.as_nanos();
                    if period == 0 {
                        return Err(timer_error(
                            &TIMER_ERROR_INVALID_PROPERTY,
                            "timer: fixed-rate Period must be positive",
                        ));
                    }
                    let lag = now.saturating_sub(next_deadline).as_nanos();
                    let missed = ((lag - 1) / period) + 1;
                    let remaining = config.tasks - scheduled;
                    if missed >= remaining as u128 {
                        break;
                    }
                    let advance = period.checked_mul(missed).ok_or_else(|| {
                        timer_error(
                            &TIMER_ERROR_INVALID_PROPERTY,
                            "timer: schedule deadline overflow",
                        )
                    })?;
                    let seconds = u64::try_from(advance / 1_000_000_000).map_err(|_| {
                        timer_error(
                            &TIMER_ERROR_INVALID_PROPERTY,
                            "timer: schedule deadline overflow",
                        )
                    })?;
                    next_deadline = next_deadline
                        .checked_add(Duration::new(seconds, (advance % 1_000_000_000) as u32))
                        .ok_or_else(|| {
                            timer_error(
                                &TIMER_ERROR_INVALID_PROPERTY,
                                "timer: schedule deadline overflow",
                            )
                        })?;
                    scheduled += missed as usize;
                }
                TimerBusyMode::Error => {
                    return Err(timer_error(
                        &TIMER_ERROR_CALLBACK,
                        "timer: fixed-rate callback queue is busy",
                    ));
                }
                TimerBusyMode::Queue => {}
            }
        }
        sleep_until(
            runtime,
            next_deadline.saturating_sub(timing.monotonic_elapsed()),
        )
        .await?;
    }
    Ok(())
}

fn still_running(runtime: &RuntimeContext, handle: &HandleRef, generation: u64) -> bool {
    runtime
        .timer_session()
        .borrow()
        .record(handle)
        .is_some_and(|entry| entry.generation == generation && entry.phase == TimerPhase::Running)
}

async fn sleep_until(runtime: &RuntimeContext, duration: Duration) -> BuiltinResult<()> {
    runtime
        .cpu_cooperation()
        .await_cancellable(runtime.service_ports().timing().sleep(duration))
        .await
        .map_err(crate::execution::ExecutionServiceError::into_runtime_error)??;
    Ok(())
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
    set_property(handle, "Running", Value::String("off".into()))?;
    if call_stop && claimed {
        run_callback(runtime, handle, "StopFcn", "StopFcn").await?;
    }
    Ok(())
}

pub(super) async fn wait_one_timer(
    runtime: &RuntimeContext,
    handle: &HandleRef,
) -> BuiltinResult<()> {
    ensure_valid_timer(handle)?;
    let lease = runtime
        .timer_session()
        .borrow()
        .record(handle)
        .ok_or_else(|| admission_error(TimerAdmissionError::ForeignOrDeleted))?
        .lease
        .clone();
    if let Some(lease) = lease {
        if lease.join().await == RuntimeLocalTaskOutcome::Cancelled {
            let _ = runtime.timer_session().borrow_mut().stop(handle);
            let _ = set_property(handle, "Running", Value::String("off".into()));
            return Err(timer_error(
                &TIMER_ERROR_CALLBACK,
                "timer: callback task was cancelled before completion",
            ));
        }
    }
    if let Some(error) = runtime
        .timer_session()
        .borrow_mut()
        .take_terminal_error(handle)
    {
        return Err(error);
    }
    Ok(())
}
