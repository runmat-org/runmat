use crate::{RunnerError, RunnerResult};

pub fn local_lanes(requested: usize, host_max: usize, job_count: usize) -> RunnerResult<usize> {
    if requested == 0 || host_max == 0 {
        return Err(RunnerError::InvalidConfiguration(
            "local worker counts must be greater than zero".into(),
        ));
    }
    Ok(requested.min(host_max).min(job_count.max(1)))
}
