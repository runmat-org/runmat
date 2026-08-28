use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EscalationPolicy {
    pub graceful_millis: u64,
    pub terminate_millis: u64,
}

impl Default for EscalationPolicy {
    fn default() -> Self {
        Self {
            graceful_millis: 2_000,
            terminate_millis: 10_000,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CancellationEscalation {
    Request,
    Terminate,
    Fence,
}

impl EscalationPolicy {
    pub fn level(self, requested_at: u64, now: u64) -> CancellationEscalation {
        let elapsed = now.saturating_sub(requested_at);
        if elapsed >= self.terminate_millis {
            CancellationEscalation::Fence
        } else if elapsed >= self.graceful_millis {
            CancellationEscalation::Terminate
        } else {
            CancellationEscalation::Request
        }
    }
}
