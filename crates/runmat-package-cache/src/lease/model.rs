use runmat_package::ContentDigest;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;
use std::str::FromStr;

macro_rules! string_id {
    ($name:ident, $label:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, &'static str> {
                let value = value.into();
                if value.is_empty() || value.len() > 256 || value.chars().any(char::is_whitespace) {
                    return Err(concat!($label, " must be 1-256 non-whitespace characters"));
                }
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }

        impl FromStr for $name {
            type Err = &'static str;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::new(value)
            }
        }
    };
}

string_id!(LeaseId, "lease id");
string_id!(LeaseOwner, "lease owner");

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lease {
    pub id: LeaseId,
    pub owner: LeaseOwner,
    pub objects: BTreeSet<ContentDigest>,
    pub acquired_at_ms: u64,
    pub expires_at_ms: u64,
    pub generation: u64,
}

impl Lease {
    pub fn is_active_at(&self, now_ms: u64) -> bool {
        self.expires_at_ms > now_ms
    }
}
