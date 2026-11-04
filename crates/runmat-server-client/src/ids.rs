//! Public identifiers the client exchanges with the RunMat server API.
//! The server owns the complete identifier namespace; this module carries only
//! the identifiers the open-source client parses and sends.
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IdError {
    #[error("invalid id prefix")]
    InvalidPrefix,
    #[error("invalid id suffix")]
    InvalidSuffix,
}

fn is_valid_suffix(suffix: &str) -> bool {
    suffix.len() == 32
        && suffix
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

macro_rules! prefixed_id {
    ($name:ident, $prefix:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(try_from = "String", into = "String")]
        pub struct $name(String);

        impl TryFrom<String> for $name {
            type Error = IdError;
            fn try_from(value: String) -> Result<Self, Self::Error> {
                Self::validate(&value)?;
                Ok(Self(value))
            }
        }

        impl From<$name> for String {
            fn from(value: $name) -> Self {
                value.0
            }
        }

        impl $name {
            pub const PREFIX: &'static str = $prefix;

            pub fn parse(value: &str) -> Result<Self, IdError> {
                Self::validate(value)?;
                Ok(Self(value.to_string()))
            }

            fn validate(value: &str) -> Result<(), IdError> {
                if !value.starts_with(Self::PREFIX) {
                    return Err(IdError::InvalidPrefix);
                }
                let suffix = &value[Self::PREFIX.len()..];
                if !is_valid_suffix(suffix) {
                    return Err(IdError::InvalidSuffix);
                }
                Ok(())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

            pub fn from_uuid(value: uuid::Uuid) -> Self {
                Self(format!("{}{}", Self::PREFIX, value.simple()))
            }

            pub fn as_uuid(&self) -> Result<uuid::Uuid, IdError> {
                let suffix = self
                    .0
                    .strip_prefix(Self::PREFIX)
                    .ok_or(IdError::InvalidPrefix)?;
                uuid::Uuid::parse_str(suffix).map_err(|_| IdError::InvalidSuffix)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl FromStr for $name {
            type Err = IdError;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Self::parse(s)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }
    };
}

prefixed_id!(OrgId, "org_");
prefixed_id!(ProjectId, "proj_");
prefixed_id!(ClusterId, "cls_");
prefixed_id!(ClusterNodeId, "node_");
prefixed_id!(FsVersionId, "fsv_");
prefixed_id!(SnapshotId, "snap_");
prefixed_id!(SimulationRunId, "run_");
prefixed_id!(SimulationLeaseId, "simlease_");
prefixed_id!(SimulationReceiptId, "simreceipt_");
prefixed_id!(ExecutionDriverLeaseId, "drv_");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_requires_prefix_and_lowercase_hex_suffix() {
        let uuid = uuid::Uuid::from_u128(0x0123_4567_89ab_cdef_0123_4567_89ab_cdef);
        let project = ProjectId::from_uuid(uuid);
        assert_eq!(project.as_str(), "proj_0123456789abcdef0123456789abcdef");
        assert_eq!(ProjectId::parse(project.as_str()), Ok(project.clone()));
        assert_eq!(project.as_uuid(), Ok(uuid));
        assert_eq!(
            ProjectId::parse("org_0123456789abcdef0123456789abcdef"),
            Err(IdError::InvalidPrefix)
        );
        assert_eq!(
            ProjectId::parse("proj_0123456789ABCDEF0123456789abcdef"),
            Err(IdError::InvalidSuffix)
        );
        let cluster = ClusterId::from_uuid(uuid);
        let encoded = serde_json::to_string(&cluster).unwrap();
        assert_eq!(
            serde_json::from_str::<ClusterId>(&encoded).unwrap(),
            cluster
        );
        for invalid in [
            "node_0123456789abcdef0123456789abcdef",
            "cls_0123456789ABCDEF0123456789abcdef",
            "cls_short",
        ] {
            assert!(
                serde_json::from_value::<ClusterId>(serde_json::Value::String(invalid.into()))
                    .is_err()
            );
        }
    }
}
