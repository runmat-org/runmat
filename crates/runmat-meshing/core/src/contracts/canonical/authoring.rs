use runmat_geometry_core::GeometryDefinitionRef;
use serde::{Deserialize, Serialize};

use super::{MeshingContractError, MeshingRequestSettings};

/// An authored mesh request before exact geometry admission supplies its tolerance policy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshDefinition {
    pub module_path: String,
    pub mesh_id: String,
    pub geometry: GeometryDefinitionRef,
    pub settings: MeshingRequestSettings,
}

impl MeshDefinition {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        token("mesh module path", &self.module_path)?;
        token("mesh id", &self.mesh_id)?;
        token("geometry module path", &self.geometry.module_path)?;
        token("geometry id", &self.geometry.geometry_id)?;
        self.settings.validate()
    }
}

fn token(field: &str, value: &str) -> Result<(), MeshingContractError> {
    if value.trim().is_empty() {
        Err(MeshingContractError::invalid(field, "must not be empty"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_validates_authored_settings_without_geometry_tolerance() {
        MeshDefinition {
            module_path: "project/main.fea".into(),
            mesh_id: "bracket".into(),
            geometry: GeometryDefinitionRef {
                module_path: "project/geometry.fea".into(),
                geometry_id: "bracket".into(),
            },
            settings: MeshingRequestSettings::default(),
        }
        .validate()
        .unwrap();
    }
}
