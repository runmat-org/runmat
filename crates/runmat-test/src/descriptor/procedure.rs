use serde::{Deserialize, Serialize};

use super::SourceDescriptor;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProcedureDescriptor {
    pub semantic_path: String,
    pub display_name: String,
    pub kind: ProcedureKind,
    pub source: SourceDescriptor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcedureKind {
    ScriptSection,
    Function,
    Method,
    SuiteFactory,
    Fixture,
    Teardown,
}
