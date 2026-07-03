use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::StageEvidence;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolveReadinessReport {
    pub ready: bool,
    #[serde(default)]
    pub evidence: Vec<StageEvidence>,
    #[serde(default)]
    pub failure_counts: BTreeMap<String, usize>,
}
