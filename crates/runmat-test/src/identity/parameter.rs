use serde::{Deserialize, Serialize};

use super::algorithm;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ParameterId(String);

impl ParameterId {
    pub fn derive(name: &str, normalized_value_identity: &str) -> Self {
        Self(algorithm::digest(
            "parameter",
            &[name, normalized_value_identity],
        ))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}
