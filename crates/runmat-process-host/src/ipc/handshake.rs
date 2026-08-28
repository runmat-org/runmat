use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{HostCapability, ProcessHostError, ProcessHostResult};

use super::FrameLimits;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostHandshake {
    pub protocol: String,
    pub schema_version: u16,
    pub min_schema_version: u16,
    pub max_message_bytes: u32,
    pub capabilities: BTreeSet<HostCapability>,
}

impl HostHandshake {
    pub fn new(protocol: impl Into<String>, schema_version: u16, max_message_bytes: u32) -> Self {
        Self {
            protocol: protocol.into(),
            schema_version,
            min_schema_version: schema_version,
            max_message_bytes,
            capabilities: BTreeSet::new(),
        }
    }
}

pub fn negotiate_handshake(
    local: &HostHandshake,
    remote: &HostHandshake,
) -> ProcessHostResult<FrameLimits> {
    if local.protocol != remote.protocol {
        return Err(ProcessHostError::Protocol(format!(
            "IPC protocol mismatch: local '{}' and remote '{}'",
            local.protocol, remote.protocol
        )));
    }
    if local.schema_version < remote.min_schema_version
        || remote.schema_version < local.min_schema_version
    {
        return Err(ProcessHostError::Protocol(
            "IPC schema ranges do not overlap".into(),
        ));
    }
    FrameLimits {
        max_message_bytes: local.max_message_bytes.min(remote.max_message_bytes),
    }
    .validate()
}
