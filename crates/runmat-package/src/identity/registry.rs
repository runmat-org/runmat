use crate::policy::{validate_canonical_segment, REGISTRY_SEGMENT_MAX_LEN};
use crate::IdentityError;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;
use url::Url;

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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct RegistryOrigin(String);

impl RegistryOrigin {
    pub fn new(value: &str) -> Result<Self, IdentityError> {
        let mut url =
            Url::parse(value).map_err(|_| invalid_source(value, "must be an HTTPS URL"))?;
        if url.scheme() != "https"
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
            || !matches!(url.path(), "" | "/")
        {
            return Err(invalid_source(
                value,
                "must be a credential-free HTTPS origin without a path, query, or fragment",
            ));
        }
        let host = url
            .host_str()
            .ok_or_else(|| invalid_source(value, "must include a host"))?
            .to_ascii_lowercase();
        url.set_host(Some(&host))
            .map_err(|_| invalid_source(value, "contains an invalid host"))?;
        if url.port_or_known_default() == Some(443) {
            url.set_port(None)
                .map_err(|_| invalid_source(value, "contains an invalid port"))?;
        }
        url.set_path("");
        Ok(Self(url.to_string().trim_end_matches('/').to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for RegistryOrigin {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl FromStr for RegistryOrigin {
    type Err = IdentityError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

impl TryFrom<String> for RegistryOrigin {
    type Error = IdentityError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(&value)
    }
}

impl From<RegistryOrigin> for String {
    fn from(value: RegistryOrigin) -> Self {
        value.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct RegistryReleaseId(String);

impl RegistryReleaseId {
    pub fn new(value: impl Into<String>) -> Result<Self, IdentityError> {
        let value = value.into();
        let suffix = value
            .strip_prefix("rel_")
            .ok_or_else(|| invalid_source(&value, "release ID must start with `rel_`"))?;
        if suffix.len() != 32
            || suffix
                .bytes()
                .any(|byte| !byte.is_ascii_hexdigit() || byte.is_ascii_uppercase())
        {
            return Err(invalid_source(
                &value,
                "release ID must contain 32 lowercase hexadecimal digits",
            ));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for RegistryReleaseId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl FromStr for RegistryReleaseId {
    type Err = IdentityError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

impl TryFrom<String> for RegistryReleaseId {
    type Error = IdentityError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<RegistryReleaseId> for String {
    fn from(value: RegistryReleaseId) -> Self {
        value.0
    }
}

fn invalid_source(value: &str, reason: &'static str) -> IdentityError {
    IdentityError::InvalidRegistrySource {
        value: value.to_string(),
        reason,
    }
}
