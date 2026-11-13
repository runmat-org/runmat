//! Native persistence evidence is separate from the scheduler's execution verdict.
use runmat_filesystem::{AtomicReplaceOutcome, PublicationDurability};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "persistence", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeCheckpointEvidence {
    Durable,
    NotPublished { diagnostic: String },
    DurabilityIndeterminate { diagnostic: String },
}

impl NativeCheckpointEvidence {
    pub(crate) fn from_publication(outcome: AtomicReplaceOutcome) -> Self {
        match outcome.durability {
            PublicationDurability::Durable => Self::Durable,
            PublicationDurability::AtomicVisibility => Self::DurabilityIndeterminate {
                diagnostic: outcome.durability_diagnostic.unwrap_or_else(|| {
                    "provider established atomic visibility without confirming crash durability"
                        .into()
                }),
            },
        }
    }

    pub(crate) fn delivery_error(&self) -> Option<String> {
        match self {
            Self::Durable => None,
            Self::NotPublished { diagnostic } => Some(format!("native completion checkpoint was not published: {diagnostic}")),
            Self::DurabilityIndeterminate { diagnostic } => Some(format!("native completion checkpoint is visible but durability is indeterminate: {diagnostic}")),
        }
    }
}
