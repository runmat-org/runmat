use serde::{Deserialize, Serialize};

use crate::driver::{Driver, DriverSnapshot};
use crate::{RunnerError, RunnerResult};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DriverCheckpoint<S = DriverSnapshot> {
    pub schema_version: u16,
    pub generation: u64,
    pub snapshot: S,
}

impl<S: std::borrow::Borrow<DriverSnapshot>> DriverCheckpoint<S> {
    /// Validate portable checkpoint schema and its complete original scheduler history.
    /// Hosts must not duplicate these format rules in their transport wrappers.
    pub fn validate(&self) -> RunnerResult<()> {
        self.validate_schema()?;
        self.snapshot.borrow().validate()
    }

    fn validate_schema(&self) -> RunnerResult<()> {
        if self.schema_version != 2 {
            return Err(RunnerError::Invalid(
                "unsupported driver checkpoint schema".into(),
            ));
        }
        Ok(())
    }

    pub fn new(generation: u64, snapshot: S) -> Self {
        Self {
            schema_version: 2,
            generation,
            snapshot,
        }
    }
}

impl DriverCheckpoint {
    /// Continue the same persisted driver authority under exclusive host ownership.
    ///
    /// The host must hold its authority lock while reading the latest journal and
    /// throughout subsequent execution. No replacement authority may have advanced
    /// the fence. Detached providers may publish reports, but cannot mutate that
    /// journal. This preserves assigned attempts; replacement recovery must use
    /// `restore` or `restore_at_fence`, which retain unresolved original work
    /// and its reservations until its owning backend proves retirement.
    pub fn continue_under_exclusive_ownership(self) -> RunnerResult<Driver> {
        self.validate()?;
        Driver::from_snapshot(self.snapshot)
    }

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
}

#[cfg(test)]
mod tests;
