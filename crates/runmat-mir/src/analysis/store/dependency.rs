use runmat_types::InvalidationCause;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisDependency {
    pub identity: String,
    pub revision: String,
    pub invalidates: Vec<InvalidationCause>,
}
