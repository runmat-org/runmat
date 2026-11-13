use serde::{Deserialize, Serialize};

use super::MeshingContractError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CancellationPolicy {
    /// Responsiveness target reported by telemetry when a checkpoint gap exceeds it. It is not a
    /// failure limit: latency depends on machine load, not on the algorithm.
    #[serde(with = "runmat_types::wire_integer")]
    pub maximum_checkpoint_latency_ms: u64,
    /// Enforced responsiveness contract: search work admitted between two cancellation probes.
    #[serde(with = "runmat_types::wire_integer")]
    pub maximum_work_units_between_checks: u64,
}

impl CancellationPolicy {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        if self.maximum_checkpoint_latency_ms == 0 || self.maximum_work_units_between_checks == 0 {
            return Err(MeshingContractError::invalid(
                "cancellation policy",
                "latency and work interval must be non-zero",
            ));
        }
        Ok(())
    }
}

pub trait MeshingCancellationSignal: Send + Sync {
    /// Retain the probe site for execution-owned responsiveness diagnostics.
    #[track_caller]
    fn is_cancelled(&self) -> bool;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NeverCancelled;

impl MeshingCancellationSignal for NeverCancelled {
    fn is_cancelled(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::MeshingCancellationSignal;
    use std::sync::Mutex;

    struct CallerCapture(Mutex<Option<(&'static str, u32)>>);
    impl MeshingCancellationSignal for CallerCapture {
        fn is_cancelled(&self) -> bool {
            let caller = std::panic::Location::caller();
            *self.0.lock().unwrap() = Some((caller.file(), caller.line()));
            false
        }
    }

    #[test]
    fn dynamic_cancellation_probe_retains_its_caller() {
        let capture = CallerCapture(Mutex::new(None));
        let signal: &dyn MeshingCancellationSignal = &capture;
        let expected_line = line!() + 1;
        assert!(!signal.is_cancelled());
        assert_eq!(*capture.0.lock().unwrap(), Some((file!(), expected_line)));
    }
}
