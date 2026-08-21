use serde::{Deserialize, Serialize};

use super::algorithm;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RunId(String);

impl RunId {
    pub fn derive(program_revision: &str, invocation_identity: &str) -> Self {
        Self(algorithm::digest(
            "run",
            &[program_revision, invocation_identity],
        ))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}
