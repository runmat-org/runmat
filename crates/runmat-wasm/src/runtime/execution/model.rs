use serde::Deserialize;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BrowserWorkerTopology {
    Coordinator,
    Flat,
    Serial,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BrowserExecutionCapabilities {
    pub(crate) topology: BrowserWorkerTopology,
    pub(crate) max_workers: u32,
    /// Owned-builtin launches certify physical exit on fulfillment; rejected
    /// Promises require a task-bound cancelOwned acknowledgment.
    #[serde(default)]
    pub(crate) owned_builtin_work: bool,
    /// Selected Direct/Graph programs require an exact Runtime-owned
    /// invocation-retirement terminal, separate from literal owned-builtin exit.
    #[serde(default)]
    pub(crate) selected_program_work: bool,
}

impl Default for BrowserExecutionCapabilities {
    fn default() -> Self {
        Self {
            topology: BrowserWorkerTopology::Serial,
            max_workers: 1,
            owned_builtin_work: false,
            selected_program_work: false,
        }
    }
}

impl BrowserExecutionCapabilities {
    pub(crate) fn validate(self) -> Result<Self, String> {
        if self.max_workers == 0 || self.max_workers > 256 {
            return Err("browser execution maxWorkers must be between 1 and 256".into());
        }
        if self.topology == BrowserWorkerTopology::Serial && self.max_workers != 1 {
            return Err("the serial browser execution topology requires maxWorkers = 1".into());
        }
        if self.owned_builtin_work && !self.has_worker_isolation() {
            return Err("ownedBuiltinWork requires isolated browser workers".into());
        }
        if self.selected_program_work && !self.has_worker_isolation() {
            return Err("selectedProgramWork requires isolated browser workers".into());
        }
        Ok(self)
    }

    pub(crate) fn has_worker_isolation(self) -> bool {
        self.topology != BrowserWorkerTopology::Serial
    }
}
