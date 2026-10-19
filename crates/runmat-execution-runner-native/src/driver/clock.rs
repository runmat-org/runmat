//! Feed time into the existing portable driver while native workers execute.
//! Scheduling, deadlines and cancellation policy remain in that shared owner.
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::LocalDriver;
use crate::{NativeExecutionError, NativeExecutionResult};
use runmat_execution_runner::DriverCommand;

pub(super) struct DriverClock {
    unix_millis: u64,
    started: Instant,
}

impl DriverClock {
    pub(super) fn new() -> NativeExecutionResult<Self> {
        let started = Instant::now();
        let elapsed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| NativeExecutionError::Configuration(error.to_string()))?;
        let unix_millis = u64::try_from(elapsed.as_millis())
            .map_err(|error| NativeExecutionError::Configuration(error.to_string()))?;
        Ok(Self {
            unix_millis,
            started,
        })
    }

    pub(super) fn now_millis(&self) -> u64 {
        // Anchor Unix deadlines once; wall-clock adjustments cannot reverse elapsed
        // reservations. Saturation expires remaining work at the clock's limit.
        self.unix_millis
            .saturating_add(u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX))
    }
}

pub(super) fn start(local: &Arc<LocalDriver>) -> NativeExecutionResult<()> {
    let weak = Arc::downgrade(local);
    std::thread::Builder::new()
        .name("native-driver-clock".into())
        .spawn(move || loop {
            std::thread::sleep(Duration::from_millis(10));
            let Some(local) = weak.upgrade() else {
                break;
            };
            let actions = {
                let mut driver = local.driver.lock().expect("local driver poisoned");
                let now_millis = local.clock.now_millis().max(driver.now_millis());
                driver
                    .handle(DriverCommand::Tick { now_millis })
                    .expect("native clock feeds monotonic ticks into an admitted driver")
            };
            LocalDriver::dispatch(local, actions);
        })
        .map_err(|error| NativeExecutionError::Configuration(error.to_string()))?;
    Ok(())
}
