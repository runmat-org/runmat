use crate::{RunnerError, RunnerResult};

pub fn next_driver_fence(current: u64) -> RunnerResult<u64> {
    current
        .checked_add(1)
        .ok_or_else(|| RunnerError::Invalid("driver fence is exhausted".into()))
}
