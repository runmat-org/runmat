//! Authored identifiers the kernel echoes back in history tags.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A sketch entity's authored name within its sketch (`l1`, `a1`). Compared as a whole value
/// and never parsed.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SketchEntityId(pub String);

impl SketchEntityId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SketchEntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for SketchEntityId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}
