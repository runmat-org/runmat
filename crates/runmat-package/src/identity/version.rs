use crate::IdentityError;
use semver::Version;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PackageVersion(Version);

impl PackageVersion {
    pub fn new(version: Version) -> Self {
        Self(version)
    }

    pub fn as_semver(&self) -> &Version {
        &self.0
    }
}

impl Display for PackageVersion {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, formatter)
    }
}

impl FromStr for PackageVersion {
    type Err = IdentityError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Version::parse(value)
            .map(Self)
            .map_err(|error| IdentityError::InvalidVersion {
                value: value.to_string(),
                reason: error.to_string(),
            })
    }
}
