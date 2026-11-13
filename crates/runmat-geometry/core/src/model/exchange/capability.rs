use serde::{Deserialize, Serialize};

use super::{invalid, token};
use crate::GeometryContractError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ap242Capability {
    ProductStructure,
    PersistentIds,
    Parameters,
    Constraints,
    Features,
    ProductManufacturingInformation,
    UserAttributes,
    Materials,
    EngineeringProperties,
    Composites,
    ValidationProperties,
    Regeneration,
    ChangeSet,
    Export,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ap242CapabilityDisposition {
    Absent,
    PresentUnsupported,
    ReadOnly,
    ReadWrite,
    Regenerable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ap242CapabilityRecord {
    pub capability: Ap242Capability,
    pub disposition: Ap242CapabilityDisposition,
    #[serde(with = "runmat_types::wire_integer::vec")]
    pub evidence_entity_ids: Vec<u64>,
    pub detail: String,
}

impl Ap242CapabilityRecord {
    pub(super) fn validate(&self) -> Result<(), GeometryContractError> {
        token("AP242 capability detail", &self.detail)?;
        if self.evidence_entity_ids.contains(&0) {
            return Err(invalid(
                "AP242 capability evidence",
                "source entity ids must be nonzero",
            ));
        }
        if self.disposition == Ap242CapabilityDisposition::Absent
            && !self.evidence_entity_ids.is_empty()
        {
            return Err(invalid(
                "AP242 capability evidence",
                "an absent capability cannot cite source entities",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ap242PersistentId {
    pub persistent_id: String,
    #[serde(with = "runmat_types::wire_integer")]
    pub source_entity_id: u64,
    #[serde(default)]
    #[serde(with = "runmat_types::wire_integer::option")]
    pub owner_entity_id: Option<u64>,
    pub role: Ap242PersistentIdRole,
}

impl Ap242PersistentId {
    pub(super) fn validate(&self) -> Result<(), GeometryContractError> {
        token("AP242 persistent id", &self.persistent_id)?;
        if self.source_entity_id == 0 || self.owner_entity_id == Some(0) {
            return Err(invalid(
                "AP242 persistent id source",
                "source identities must be nonzero",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ap242PersistentIdRole {
    Product,
    ProductDefinition,
    Occurrence,
    ShapeAspect,
    RepresentationItem,
    Property,
    Material,
}
