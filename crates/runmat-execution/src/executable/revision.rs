use serde::{Deserialize, Serialize};

use crate::{ContractError, Digest};
use runmat_types::{
    INTEROP_MANIFEST_SCHEMA_VERSION, PARALLEL_MANIFEST_SCHEMA_VERSION,
    REGION_CONTRACT_SCHEMA_VERSION,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutableComponentRevisions {
    pub catalog_schema: u16,
    pub catalog_fingerprint: Digest,
    pub contract_schema: u16,
    pub contract_fingerprint: Digest,
    pub analysis_schema: u16,
    pub mir_schema: u16,
    pub bytecode_schema: u16,
    pub vm_layout_schema: u16,
    pub function_registry_schema: u16,
    pub source_map_schema: u16,
    pub region_schema: u16,
    pub interop_schema: u16,
    pub parallel_schema: u16,
}

impl ExecutableComponentRevisions {
    pub(crate) fn validate(&self) -> Result<(), ContractError> {
        for (field, version) in [
            ("executable.revisions.catalog_schema", self.catalog_schema),
            ("executable.revisions.contract_schema", self.contract_schema),
            ("executable.revisions.analysis_schema", self.analysis_schema),
            ("executable.revisions.mir_schema", self.mir_schema),
            ("executable.revisions.bytecode_schema", self.bytecode_schema),
            (
                "executable.revisions.vm_layout_schema",
                self.vm_layout_schema,
            ),
            (
                "executable.revisions.function_registry_schema",
                self.function_registry_schema,
            ),
            (
                "executable.revisions.source_map_schema",
                self.source_map_schema,
            ),
        ] {
            if version == 0 {
                return Err(ContractError::invalid(field, "version must be non-zero"));
            }
        }
        for (field, actual, supported) in [
            (
                "executable.revisions.region_schema",
                self.region_schema,
                REGION_CONTRACT_SCHEMA_VERSION,
            ),
            (
                "executable.revisions.interop_schema",
                self.interop_schema,
                INTEROP_MANIFEST_SCHEMA_VERSION,
            ),
            (
                "executable.revisions.parallel_schema",
                self.parallel_schema,
                PARALLEL_MANIFEST_SCHEMA_VERSION,
            ),
        ] {
            if actual != supported {
                return Err(ContractError::invalid(
                    field,
                    format!("unsupported version {actual}; expected {supported}"),
                ));
            }
        }
        Ok(())
    }
}
