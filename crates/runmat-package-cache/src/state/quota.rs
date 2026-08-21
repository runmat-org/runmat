use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QuotaPressure {
    Normal,
    Elevated,
    Critical,
    Exhausted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuotaRecord {
    pub limit_bytes: u64,
    pub observed_bytes: u64,
    pub pressure: QuotaPressure,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    pub updated_at_ms: u64,
}

impl QuotaRecord {
    pub fn available_bytes(&self) -> u64 {
        self.limit_bytes.saturating_sub(self.observed_bytes)
    }
}
