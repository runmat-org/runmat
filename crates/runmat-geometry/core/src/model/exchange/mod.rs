mod capability;
mod change_set;
mod parameter;
mod property;

pub use capability::*;
pub use change_set::*;
pub use parameter::*;
pub use property::*;

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use super::GeometryContractError;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Ap242ExchangeInventory {
    pub schemas: Vec<String>,
    pub capabilities: Vec<Ap242CapabilityRecord>,
    pub persistent_ids: Vec<Ap242PersistentId>,
    pub parameters: Vec<Ap242Parameter>,
    pub engineering_properties: Vec<Ap242EngineeringProperty>,
    pub materials: Vec<Ap242MaterialCandidate>,
    pub associations: Vec<Ap242Association>,
    pub unsupported: Vec<Ap242UnsupportedConcept>,
}

impl Ap242ExchangeInventory {
    pub fn validate(&self) -> Result<(), GeometryContractError> {
        unique_tokens("AP242 schemas", self.schemas.iter().map(String::as_str))?;
        unique_by("AP242 capabilities", &self.capabilities, |value| {
            format!("{:?}", value.capability)
        })?;
        unique_by("AP242 persistent ids", &self.persistent_ids, |value| {
            value.persistent_id.clone()
        })?;
        unique_by("AP242 parameters", &self.parameters, |value| {
            value.parameter_id.clone()
        })?;
        unique_by(
            "AP242 engineering properties",
            &self.engineering_properties,
            |value| value.property_id.clone(),
        )?;
        unique_by("AP242 materials", &self.materials, |value| {
            value.material_id.clone()
        })?;
        unique_by("AP242 associations", &self.associations, |value| {
            value.association_id.clone()
        })?;
        for value in &self.capabilities {
            value.validate()?;
        }
        for value in &self.persistent_ids {
            value.validate()?;
        }
        for value in &self.parameters {
            value.validate()?;
        }
        for value in &self.engineering_properties {
            value.validate()?;
        }
        for value in &self.materials {
            value.validate(self)?;
        }
        for value in &self.associations {
            value.validate(self)?;
        }
        for value in &self.unsupported {
            value.validate()?;
        }
        Ok(())
    }

    pub fn capability(&self, capability: Ap242Capability) -> Ap242CapabilityDisposition {
        self.capabilities
            .iter()
            .find(|record| record.capability == capability)
            .map(|record| record.disposition)
            .unwrap_or(Ap242CapabilityDisposition::Absent)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ap242UnsupportedConcept {
    pub entity_type: String,
    #[serde(with = "runmat_types::wire_integer::vec")]
    pub entity_ids: Vec<u64>,
    pub reason: String,
}

impl Ap242UnsupportedConcept {
    fn validate(&self) -> Result<(), GeometryContractError> {
        token("AP242 unsupported entity type", &self.entity_type)?;
        token("AP242 unsupported reason", &self.reason)?;
        if self.entity_ids.is_empty() || self.entity_ids.contains(&0) {
            return Err(invalid(
                "AP242 unsupported entity ids",
                "must contain nonzero source entity ids",
            ));
        }
        Ok(())
    }
}

pub(super) fn token(field: &str, value: &str) -> Result<(), GeometryContractError> {
    if value.trim().is_empty() {
        return Err(invalid(field, "must not be empty"));
    }
    Ok(())
}

pub(super) fn invalid(field: &str, reason: &str) -> GeometryContractError {
    GeometryContractError::invalid(field, reason)
}

fn unique_tokens<'a>(
    field: &str,
    values: impl Iterator<Item = &'a str>,
) -> Result<(), GeometryContractError> {
    let mut seen = BTreeSet::new();
    for value in values {
        token(field, value)?;
        if !seen.insert(value) {
            return Err(invalid(field, "values must be unique"));
        }
    }
    Ok(())
}

fn unique_by<T>(
    field: &str,
    values: &[T],
    key: impl Fn(&T) -> String,
) -> Result<(), GeometryContractError> {
    let mut seen = BTreeSet::new();
    if values.iter().any(|value| !seen.insert(key(value))) {
        return Err(invalid(field, "identities must be unique"));
    }
    Ok(())
}
