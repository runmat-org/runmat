use crate::policy::{validate_canonical_segment, REGISTRY_SEGMENT_MAX_LEN};
use crate::IdentityError;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct RegistryId(String);

impl RegistryId {
    pub fn new(value: impl Into<String>) -> Result<Self, IdentityError> {
        let value = value.into();
        validate_canonical_segment(&value, "registry name", REGISTRY_SEGMENT_MAX_LEN)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for RegistryId {
    fn default() -> Self {
        Self("default".to_string())
    }
}

impl Display for RegistryId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl FromStr for RegistryId {
    type Err = IdentityError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

impl TryFrom<String> for RegistryId {
    type Error = IdentityError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<RegistryId> for String {
    fn from(value: RegistryId) -> Self {
        value.0
    }
}
