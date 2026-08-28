use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostCapability {
    ProcessIsolation,
    CapturedStderr,
    StdioIpc,
    SharedMemory,
}
