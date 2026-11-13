use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;

macro_rules! operation_identity {
    ($name:ident, $label:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, OperationIdentityError> {
                let value = value.into();
                if value.trim().is_empty() {
                    return Err(OperationIdentityError { kind: $label });
                }
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(formatter)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Self::new(value).map_err(serde::de::Error::custom)
            }
        }
    };
}

operation_identity!(OperationIdentity, "operation");
operation_identity!(OperationOverloadIdentity, "operation overload");
operation_identity!(
    OperationCatalogExtensionIdentity,
    "operation catalog extension"
);

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{kind} identity must not be empty")]
pub struct OperationIdentityError {
    kind: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identities_reject_empty_construction_and_decoding() {
        assert!(OperationIdentity::new("  ").is_err());
        assert!(serde_json::from_str::<OperationOverloadIdentity>("\"\"").is_err());
        assert!(serde_json::from_str::<OperationCatalogExtensionIdentity>("\"  \"").is_err());
    }
}
