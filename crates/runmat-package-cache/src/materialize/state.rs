use crate::lease::LeaseId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "kebab-case")]
pub enum MaterializationState {
    Staging { attempt: String },
    Verified,
    Promoted,
    Corrupt { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterializationRecord {
    pub state: MaterializationState,
    pub lease: LeaseId,
    pub updated_at_ms: u64,
}

impl MaterializationState {
    pub fn can_transition_to(&self, next: &Self) -> bool {
        matches!(
            (self, next),
            (Self::Staging { .. }, Self::Verified | Self::Corrupt { .. })
                | (Self::Verified, Self::Promoted | Self::Corrupt { .. })
                | (Self::Corrupt { .. }, Self::Staging { .. })
        ) || self == next
    }
}
