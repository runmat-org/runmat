use serde::{Deserialize, Serialize};

use super::algorithm;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FixtureId(String);

impl FixtureId {
    pub fn derive(suite_identity: &str, semantic_fixture_path: &str) -> Self {
        Self(algorithm::digest(
            "fixture",
            &[suite_identity, semantic_fixture_path],
        ))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FixtureGroupId(String);

impl FixtureGroupId {
    pub fn derive(suite_identity: &str, shared_state_identity: &str) -> Self {
        Self(algorithm::digest(
            "fixture-group",
            &[suite_identity, shared_state_identity],
        ))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}
