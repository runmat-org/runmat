use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TestCapability {
    Filesystem,
    Network,
    Subprocess,
    NativeLibrary,
    Mex,
    Java,
    Gpu,
    SharedMemory,
    Worker,
    Custom(String),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResourceRequirement {
    pub name: String,
    pub units: u32,
    pub exclusive: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TestRequirements {
    #[serde(default)]
    pub capabilities: Vec<TestCapability>,
    #[serde(default)]
    pub resources: Vec<ResourceRequirement>,
}
