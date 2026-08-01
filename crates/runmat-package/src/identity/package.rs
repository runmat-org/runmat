use super::RegistryId;
use crate::policy::{validate_canonical_segment, PACKAGE_ALIAS_MAX_LEN, PACKAGE_SEGMENT_MAX_LEN};
use crate::IdentityError;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct CanonicalPackageId {
    registry: RegistryId,
    organization: String,
    name: String,
}

impl CanonicalPackageId {
    pub fn new(
        registry: RegistryId,
        organization: impl Into<String>,
        name: impl Into<String>,
    ) -> Result<Self, IdentityError> {
        let organization = organization.into();
        let name = name.into();
        validate_canonical_segment(
            &organization,
            "package organization",
            PACKAGE_SEGMENT_MAX_LEN,
        )?;
        validate_canonical_segment(&name, "package name", PACKAGE_SEGMENT_MAX_LEN)?;
        Ok(Self {
            registry,
            organization,
            name,
        })
    }

    pub fn registry(&self) -> &RegistryId {
        &self.registry
    }

    pub fn organization(&self) -> &str {
        &self.organization
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Display for CanonicalPackageId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}:{}/{}",
            self.registry, self.organization, self.name
        )
    }
}

impl FromStr for CanonicalPackageId {
    type Err = IdentityError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let Some((registry, package)) = value.split_once(':') else {
            return Err(invalid_package(value, "missing registry prefix"));
        };
        let Some((organization, name)) = package.split_once('/') else {
            return Err(invalid_package(value, "missing organization namespace"));
        };
        if name.contains('/') {
            return Err(invalid_package(
                value,
                "contains more than two package segments",
            ));
        }
        Self::new(registry.parse()?, organization, name)
    }
}

impl TryFrom<String> for CanonicalPackageId {
    type Error = IdentityError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

impl From<CanonicalPackageId> for String {
    fn from(value: CanonicalPackageId) -> Self {
        value.to_string()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct PackageAlias(String);

impl PackageAlias {
    pub fn new(value: impl Into<String>) -> Result<Self, IdentityError> {
        let value = value.into();
        validate_canonical_segment(&value, "dependency alias", PACKAGE_ALIAS_MAX_LEN)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for PackageAlias {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl FromStr for PackageAlias {
    type Err = IdentityError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

impl TryFrom<String> for PackageAlias {
    type Error = IdentityError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<PackageAlias> for String {
    fn from(value: PackageAlias) -> Self {
        value.0
    }
}

fn invalid_package(value: &str, reason: &'static str) -> IdentityError {
    IdentityError::InvalidName {
        kind: "canonical package ID",
        value: value.to_string(),
        reason,
    }
}
