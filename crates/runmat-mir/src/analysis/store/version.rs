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

    pub fn validate_current(&self) -> Result<(), AnalysisRevisionMismatch> {
        let current = Self::current();
        if self.schema_version != current.schema_version {
            return Err(AnalysisRevisionMismatch::StoreSchema {
                expected: current.schema_version,
                actual: self.schema_version,
            });
        }
        if self.fact_schema_major != current.fact_schema_major
            || self.fact_schema_minor > current.fact_schema_minor
        {
            return Err(AnalysisRevisionMismatch::FactSchema {
                expected_major: current.fact_schema_major,
                maximum_minor: current.fact_schema_minor,
                actual_major: self.fact_schema_major,
                actual_minor: self.fact_schema_minor,
            });
        }
        if self.catalog_schema != current.catalog_schema {
            return Err(AnalysisRevisionMismatch::CatalogSchema {
                expected: current.catalog_schema,
                actual: self.catalog_schema,
            });
        }
        if self.catalog_fingerprint != current.catalog_fingerprint {
            return Err(AnalysisRevisionMismatch::CatalogFingerprint);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnalysisRevisionMismatch {
    StoreSchema {
        expected: u16,
        actual: u16,
    },
    FactSchema {
        expected_major: u16,
        maximum_minor: u16,
        actual_major: u16,
        actual_minor: u16,
    },
    CatalogSchema {
        expected: u32,
        actual: u32,
    },
    CatalogFingerprint,
}

impl std::fmt::Display for AnalysisRevisionMismatch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StoreSchema { expected, actual } => write!(
                formatter,
                "unsupported analysis-store schema {actual}; expected {expected}"
            ),
            Self::FactSchema {
                expected_major,
                maximum_minor,
                actual_major,
                actual_minor,
            } => write!(
                formatter,
                "unsupported fact schema {actual_major}.{actual_minor}; expected {expected_major}.0 through {expected_major}.{maximum_minor}"
            ),
            Self::CatalogSchema { expected, actual } => write!(
                formatter,
                "unsupported builtin-catalog schema {actual}; expected {expected}"
            ),
            Self::CatalogFingerprint => {
                formatter.write_str("builtin-catalog fingerprint does not match this consumer")
            }
        }
    }
}

impl std::error::Error for AnalysisRevisionMismatch {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_revision_validates_and_incompatible_components_fail_closed() {
        let current = AnalysisRevision::current();
        assert_eq!(current.validate_current(), Ok(()));

        let mut future_store = current.clone();
        future_store.schema_version += 1;
        assert!(matches!(
            future_store.validate_current(),
            Err(AnalysisRevisionMismatch::StoreSchema { .. })
        ));

        let mut future_facts = current.clone();
        future_facts.fact_schema_minor += 1;
        assert!(matches!(
            future_facts.validate_current(),
            Err(AnalysisRevisionMismatch::FactSchema { .. })
        ));

        let mut future_catalog = current.clone();
        future_catalog.catalog_schema += 1;
        assert!(matches!(
            future_catalog.validate_current(),
            Err(AnalysisRevisionMismatch::CatalogSchema { .. })
        ));

        let mut other_catalog = current;
        other_catalog.catalog_fingerprint[0] ^= 1;
        assert_eq!(
            other_catalog.validate_current(),
            Err(AnalysisRevisionMismatch::CatalogFingerprint)
        );
    }
}
