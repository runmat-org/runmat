use serde::{Deserialize, Serialize};

use crate::version::PROTOCOL_VERSION;

use super::ProtocolLimits;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProtocolHandshake {
    pub protocol_version: u16,
    pub implementation: String,
    pub capabilities: Vec<WorkerCapability>,
    pub limits: ProtocolLimits,
}

impl ProtocolHandshake {
    pub fn current(implementation: impl Into<String>, capabilities: Vec<WorkerCapability>) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            implementation: implementation.into(),
            capabilities,
            limits: ProtocolLimits::default(),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerCapability {
    SessionIsolation,
    StrongIsolation,
    Coverage,
    Artifacts,
    CapturedOutput,
    Custom(String),
}
