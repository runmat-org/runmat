//! Monotonic callback scheduling and fixed-rate busy policies.

use super::*;

pub(super) async fn run_timer_callbacks(
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
    run_callback(runtime, handle, TimerCallback::Start).await?;
    if !still_running(runtime, handle, generation) {
        return Ok(());
    }
    sleep_until(runtime, config.start_delay).await?;
    let timing = runtime.service_ports().timing()?.clone();
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
            update_period_properties(runtime, handle, instant, executed)?;
        }
        last_fire = Some(fired);
        run_callback(runtime, handle, TimerCallback::Timer).await?;
        if !still_running(runtime, handle, generation) {
            return Ok(());
        }
        executed += 1;
        scheduled += 1;
        set_property(
            runtime,
            handle,
            "TasksExecuted",
            Value::Num(executed as f64),
        )?;
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

pub(super) fn still_running(runtime: &RuntimeContext, handle: &HandleRef, generation: u64) -> bool {
    runtime
        .timer_session()
        .borrow()
        .record(handle)
        .is_some_and(|entry| entry.generation == generation && entry.phase == TimerPhase::Running)
}

async fn sleep_until(runtime: &RuntimeContext, duration: Duration) -> BuiltinResult<()> {
    runtime
        .cpu_cooperation()
        .await_cancellable(runtime.service_ports().timing()?.sleep(duration))
        .await
        .map_err(crate::execution::ExecutionServiceError::into_runtime_error)??;
    Ok(())
}
