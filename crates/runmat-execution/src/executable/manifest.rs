use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::{
    ExecutableComponentRevisions, ExecutableIdentity, ExecutableOptionalSection,
    ExecutableSectionSupport,
};
use crate::{ContractError, Digest};
use runmat_types::{
    CapabilityRequirement, CapabilitySet, InteropManifest, ParallelManifest, RegionContract,
};

pub const EXECUTABLE_UNIT_SCHEMA_VERSION: u16 = crate::schema::EXECUTABLE_UNIT_SCHEMA_V1;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutableUnitManifest {
    pub schema_version: u16,
    pub identity: ExecutableIdentity,
    pub revisions: ExecutableComponentRevisions,
    pub capabilities: CapabilitySet,
    pub regions: Vec<RegionContract>,
    pub interop: InteropManifest,
    pub parallel: ParallelManifest,
    pub optional_sections: Vec<ExecutableOptionalSection>,
}

impl ExecutableUnitManifest {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != EXECUTABLE_UNIT_SCHEMA_VERSION {
            return Err(ContractError::UnsupportedSchema {
                actual: self.schema_version,
                supported: EXECUTABLE_UNIT_SCHEMA_VERSION,
            });
        }
        self.identity.validate()?;
        self.revisions.validate()?;
        if self.identity.program.catalog_fingerprint() != &self.revisions.catalog_fingerprint {
            return Err(ContractError::invalid(
                "executable.revisions.catalog_fingerprint",
                "must match the immutable program revision",
            ));
        }
        if self.regions.windows(2).any(|pair| pair[0].id >= pair[1].id) {
            return Err(ContractError::invalid(
                "executable.regions",
                "regions must be sorted and unique by identity",
            ));
        }
        for region in &self.regions {
            region.validate().map_err(|error| {
                ContractError::invalid(
                    "executable.regions",
                    format!("{}: {}", error.path, error.message),
                )
            })?;
        }
        self.interop.validate().map_err(|error| {
            ContractError::invalid(
                "executable.interop",
                format!("{}: {}", error.path, error.message),
            )
        })?;
        self.parallel.validate().map_err(|error| {
            ContractError::invalid(
                "executable.parallel",
                format!("{}: {}", error.path, error.message),
            )
        })?;
        let region_ids = self
            .regions
            .iter()
            .map(|region| region.id)
            .collect::<BTreeSet<_>>();
        if self
            .parallel
            .parfor_regions
            .iter()
            .map(|region| region.id.0)
            .chain(self.parallel.spmd_regions.iter().map(|region| region.id.0))
            .any(|region| !region_ids.contains(&region))
        {
            return Err(ContractError::invalid(
                "executable.parallel.regions",
                "every parallel construct must name a declared region contract",
            ));
        }
        if !self.interop.foreign_types.is_empty()
            && !self
                .capabilities
                .0
                .contains(&CapabilityRequirement::ForeignRuntime)
        {
            return Err(ContractError::invalid(
                "executable.capabilities",
                "foreign requirements need the foreign-runtime capability",
            ));
        }
        if (!self.parallel.parfor_regions.is_empty() || !self.parallel.spmd_regions.is_empty())
            && !self
                .capabilities
                .0
                .contains(&CapabilityRequirement::ParallelRuntime)
        {
            return Err(ContractError::invalid(
                "executable.capabilities",
                "parallel constructs need the parallel-runtime capability",
            ));
        }
        if (!self.parallel.distributed_values.is_empty() || !self.parallel.collectives.is_empty())
            && !self
                .capabilities
                .0
                .contains(&CapabilityRequirement::DistributedRuntime)
        {
            return Err(ContractError::invalid(
                "executable.capabilities",
                "distributed constructs need the distributed-runtime capability",
            ));
        }
        if self
            .optional_sections
            .windows(2)
            .any(|pair| pair[0].name >= pair[1].name)
        {
            return Err(ContractError::invalid(
                "executable.optional_sections",
                "sections must be sorted and unique by name",
            ));
        }
        for section in &self.optional_sections {
            section.validate()?;
        }
        Ok(())
    }

    pub fn validate_for(&self, support: &ExecutableSectionSupport) -> Result<(), ContractError> {
        self.validate()?;
        for section in &self.optional_sections {
            support.validate_section(section)?;
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ContractError> {
        self.validate()?;
        serde_json::to_vec(self)
            .map_err(|error| ContractError::invalid("executable.manifest", error.to_string()))
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ContractError> {
        let manifest = serde_json::from_slice::<Self>(bytes)
            .map_err(|error| ContractError::invalid("executable.manifest", error.to_string()))?;
        manifest.validate()?;
        if manifest.canonical_bytes()? != bytes {
            return Err(ContractError::invalid(
                "executable.manifest",
                "encoding is valid JSON but not canonical RunMat JSON",
            ));
        }
        Ok(manifest)
    }

    pub fn cache_key(&self) -> Result<Digest, ContractError> {
        self.canonical_bytes().map(Digest::sha256)
    }
}
