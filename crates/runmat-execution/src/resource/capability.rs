use serde::{Deserialize, Serialize};

use super::AcceleratorClass;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Selected array provenance and cyclic selected metadata graph nodes.
    GraphSelectedValuesV11,
    /// Supplied-only callable arguments in Graph V12; older workers cannot
    /// interpret this invocation contract even when they admit V11 values.
    GraphCallableArgumentsV12,
    /// Graph V13 owns digest-bound project-only execution settings.
    GraphProjectRuntimePolicyV13,
    /// Graph V14 owns exact invocation environment and bounded console controls.
    GraphInvocationControlsV14,
    ProcessIsolation,
    BrowserWorker,
    NetworkDenied,
    Accelerator(AcceleratorClass),
    Custom(String),
}
