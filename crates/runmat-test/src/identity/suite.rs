use serde::{Deserialize, Serialize};

use super::algorithm;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SuiteId(String);

impl SuiteId {
    pub fn derive(program_revision: &str, semantic_suite_path: &str) -> Self {
        Self(algorithm::digest(
            "suite",
            &[program_revision, semantic_suite_path],
        ))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}
