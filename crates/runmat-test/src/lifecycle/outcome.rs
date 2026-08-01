use serde::{Deserialize, Serialize};

use crate::result::AttemptResult;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LifecycleOutcome {
    pub attempt: AttemptResult,
    pub executed_procedures: Vec<String>,
}
