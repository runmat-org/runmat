use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccessRecord {
    pub first_seen_at_ms: u64,
    pub last_accessed_at_ms: u64,
    pub access_count: u64,
}

impl AccessRecord {
    pub fn new(now_ms: u64) -> Self {
        Self {
            first_seen_at_ms: now_ms,
            last_accessed_at_ms: now_ms,
            access_count: 1,
        }
    }

    pub fn touch(&mut self, now_ms: u64) {
        self.last_accessed_at_ms = self.last_accessed_at_ms.max(now_ms);
        self.access_count = self.access_count.saturating_add(1);
    }
}
