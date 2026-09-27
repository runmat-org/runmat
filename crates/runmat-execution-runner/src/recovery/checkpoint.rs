use serde::{Deserialize, Serialize};

use crate::driver::{Driver, DriverSnapshot};
use crate::{RunnerError, RunnerResult};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DriverCheckpoint {
    pub schema_version: u16,
    pub generation: u64,
    pub snapshot: DriverSnapshot,
}

impl DriverCheckpoint {
    /// Restore with the next available local fence.
    pub fn restore(self) -> RunnerResult<Driver> {
        self.validate_schema()?;
        super::reconcile_snapshot(self.snapshot)
    }

    /// Restore under an externally assigned, strictly newer replacement fence.
    pub fn restore_at_fence(self, replacement_fence: u64) -> RunnerResult<Driver> {
        self.validate_schema()?;
        super::reconcile_snapshot_at_fence(self.snapshot, replacement_fence)
    }

    fn validate_schema(&self) -> RunnerResult<()> {
        if self.schema_version != 1 {
            return Err(RunnerError::Invalid(
                "unsupported driver checkpoint schema".into(),
            ));
        }
        Ok(())
    }

    pub fn new(generation: u64, snapshot: DriverSnapshot) -> Self {
        Self {
            schema_version: 1,
            generation,
            snapshot,
        }
    }
}
