#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GcPolicy {
    pub now_ms: u64,
    pub retain_recent_ms: u64,
    pub target_bytes: u64,
}

impl GcPolicy {
    pub fn reclaim_to(now_ms: u64, target_bytes: u64) -> Self {
        Self {
            now_ms,
            retain_recent_ms: 0,
            target_bytes,
        }
    }
}
