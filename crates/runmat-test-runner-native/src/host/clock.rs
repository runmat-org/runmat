use std::time::{Duration, SystemTime, UNIX_EPOCH};

use runmat_test_runner::host::{Clock, PortFuture};

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeClock;

impl Clock for NativeClock {
    fn now_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX)
    }

    fn sleep_until<'a>(&'a self, deadline_ms: u64) -> PortFuture<'a, ()> {
        let delay = deadline_ms.saturating_sub(self.now_ms());
        Box::pin(tokio::time::sleep(Duration::from_millis(delay)))
    }
}
