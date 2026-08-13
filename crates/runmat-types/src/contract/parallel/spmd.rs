use crate::{LabCount, SchemaValidationError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum SpmdLabRequirement {
    Default,
    Exact {
        labs: LabCount,
    },
    Range {
        minimum: LabCount,
        maximum: LabCount,
    },
}

impl SpmdLabRequirement {
    pub(crate) fn validate(self) -> Result<(), SchemaValidationError> {
        if let Self::Range { minimum, maximum } = self {
            if maximum.0 < minimum.0 {
                return Err(SchemaValidationError::new(
                    "parallel.spmd_regions.labs",
                    "maximum lab count must not be below minimum",
                ));
            }
        }
        Ok(())
    }
}
