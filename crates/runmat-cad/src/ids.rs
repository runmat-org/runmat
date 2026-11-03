//! Identifiers used across the design layer.
//!
//! Authored identifiers (`FeatureId`, `SketchEntityId`, publish and region names) are the names
//! written in the source text. They are compared as whole values and never parsed. Renames go
//! through the lock section (`refs::lock`), which keeps the old path as an alias.

use serde::{Deserialize, Serialize};
use std::fmt;

macro_rules! name_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_owned())
            }
        }
    };
}

name_id!(
    /// A feature's authored name (`body`, `round`, `profile`). Sketches and datum features are
    /// features too, so one id space covers the whole history.
    FeatureId
);
name_id!(
    /// A sketch entity's authored name within its sketch (`l1`, `a1`).
    SketchEntityId
);
name_id!(
    /// A sketch constraint's authored name, when it has one.
    ConstraintId
);
name_id!(
    /// A part output's name within a `.part` file (`bracket`, `gasket`): `"flange.part#gasket"`.
    PartName
);
name_id!(
    /// An `input` name.
    InputName
);
name_id!(
    /// A `let` value name.
    LetName
);
name_id!(
    /// A `region` name.
    RegionName
);
name_id!(
    /// A `publish` name: part of the file's API for cross-file references.
    PublishName
);
name_id!(
    /// A `configuration` name (including `table.row` names expanded from configuration tables).
    ConfigurationName
);
name_id!(
    /// A property name from a `[properties]` schema.
    PropertyName
);
name_id!(
    /// A material reference as written (`material.aluminum`, `library.al_2024_t3`).
    MaterialRef
);
name_id!(
    /// A server-side revision id (release record or microversion). Pins carry one.
    RevisionId
);
name_id!(
    /// A partition name from `[partitions]` in `runmat.toml`.
    PartitionName
);

/// A project-relative path to a design file (`interfaces/fuselage-wing.part`). Paths are
/// compared as written; normalisation belongs to the filesystem layer.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FilePath(pub String);

impl FilePath {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for FilePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for FilePath {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

/// A 128-bit stable item id, written as 32 lowercase hex digits. The lock section mints one per
/// item path and keeps it through renames.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(into = "String", try_from = "String")]
pub struct ItemUuid(pub u128);

impl fmt::Display for ItemUuid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

impl From<ItemUuid> for String {
    fn from(value: ItemUuid) -> Self {
        value.to_string()
    }
}

impl TryFrom<String> for ItemUuid {
    type Error = String;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.len() != 32 {
            return Err(format!("item uuid must be 32 hex digits, got {:?}", value));
        }
        u128::from_str_radix(&value, 16)
            .map(ItemUuid)
            .map_err(|e| format!("item uuid {:?}: {}", value, e))
    }
}
