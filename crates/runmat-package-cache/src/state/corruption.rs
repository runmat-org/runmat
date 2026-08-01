use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorruptionRecord {
    pub detected_at_ms: u64,
    pub reason: String,
    pub occurrences: u64,
}

impl CorruptionRecord {
    pub fn new(detected_at_ms: u64, reason: impl Into<String>) -> Self {
        Self {
            detected_at_ms,
            reason: reason.into(),
            occurrences: 1,
        }
    }
}
