use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProtocolLimits {
    pub max_message_bytes: u32,
    pub max_tests_per_plan: u32,
    pub max_commands_per_invocation: u32,
    pub max_output_bytes_per_attempt: u32,
    pub max_diagnostics_per_attempt: u32,
    pub max_artifacts_per_attempt: u32,
}

impl Default for ProtocolLimits {
    fn default() -> Self {
        Self {
            max_message_bytes: 16 * 1024 * 1024,
            max_tests_per_plan: 100_000,
            max_commands_per_invocation: 256,
            max_output_bytes_per_attempt: 1024 * 1024,
            max_diagnostics_per_attempt: 10_000,
            max_artifacts_per_attempt: 1_000,
        }
    }
}
