use runmat_execution::Digest;
use runmat_package::FrozenProjectHandoff;
use serde::{Deserialize, Serialize};

use crate::{ArtifactError, ArtifactResult};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BundleCodeClosure {
    SourceProject { handoff: FrozenProjectHandoff },
    Compiled { package: CompiledPackageClosure },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledPackageClosure {
    pub schema_version: u16,
    pub graph_digest: Digest,
    pub source_digest: Digest,
    pub package_instances: Vec<String>,
}

impl CompiledPackageClosure {
    pub const SCHEMA_VERSION: u16 = 1;

    pub fn validate(&self) -> ArtifactResult<()> {
        if self.schema_version != Self::SCHEMA_VERSION
            || self.package_instances.is_empty()
            || self
                .package_instances
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self
                .package_instances
                .iter()
                .any(|identity| !valid_identity(identity))
        {
            return Err(ArtifactError::Invalid(
                "compiled package closure is not canonical".into(),
            ));
        }
        Ok(())
    }
}

fn valid_identity(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 512
        && value.is_ascii()
        && !value.chars().any(char::is_control)
}
