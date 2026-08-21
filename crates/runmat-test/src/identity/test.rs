use serde::{Deserialize, Serialize};

use super::algorithm;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TestId(String);

impl TestId {
    pub fn derive(input: &TestIdentityInput<'_>) -> Self {
        Self(algorithm::digest(
            "test",
            &[
                input.owner_identity,
                input.relative_source_identity,
                input.semantic_scheme,
                input.semantic_item_path,
                input.parameter_identity,
                input.fixture_identity,
            ],
        ))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug)]
pub struct TestIdentityInput<'a> {
    pub owner_identity: &'a str,
    pub relative_source_identity: &'a str,
    pub semantic_scheme: &'a str,
    pub semantic_item_path: &'a str,
    pub parameter_identity: &'a str,
    pub fixture_identity: &'a str,
}
