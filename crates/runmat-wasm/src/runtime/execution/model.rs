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
}

impl Default for BrowserExecutionCapabilities {
    fn default() -> Self {
        Self {
            topology: BrowserWorkerTopology::Serial,
            max_workers: 1,
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
        Ok(self)
    }

    pub(crate) fn has_worker_isolation(self) -> bool {
        self.topology != BrowserWorkerTopology::Serial
    }
}
