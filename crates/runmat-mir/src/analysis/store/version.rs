use serde::{Deserialize, Serialize};

pub const ANALYSIS_STORE_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisRevision {
    pub schema_version: u16,
    pub fact_schema_major: u16,
    pub fact_schema_minor: u16,
    pub catalog_schema: u32,
    pub catalog_fingerprint: [u8; 32],
}

impl AnalysisRevision {
    pub fn current() -> Self {
        Self {
            schema_version: ANALYSIS_STORE_SCHEMA_VERSION,
            fact_schema_major: runmat_types::RUNMAT_TYPES_SCHEMA.major,
            fact_schema_minor: runmat_types::RUNMAT_TYPES_SCHEMA.minor,
            catalog_schema: runmat_builtins::BUILTIN_CATALOG_SCHEMA,
            catalog_fingerprint: runmat_builtins::builtin_catalog_fingerprint(),
        }
    }
}
