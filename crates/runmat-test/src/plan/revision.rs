use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ProgramRevision {
    pub graph_digest: String,
    pub source_digest: String,
    pub semantic_schema: u32,
    pub compiler_schema: u32,
    pub test_config_digest: String,
}

impl ProgramRevision {
    pub fn canonical_identity(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}",
            self.graph_digest,
            self.source_digest,
            self.semantic_schema,
            self.compiler_schema,
            self.test_config_digest
        )
    }
}
