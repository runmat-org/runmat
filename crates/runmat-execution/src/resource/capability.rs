use serde::{Deserialize, Serialize};

use super::AcceleratorClass;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    ProcessIsolation,
    BrowserWorker,
    NetworkDenied,
    Accelerator(AcceleratorClass),
    Custom(String),
}
