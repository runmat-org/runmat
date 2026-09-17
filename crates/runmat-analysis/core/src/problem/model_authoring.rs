use std::collections::BTreeSet;

use runmat_geometry_core::{GeometryRegionRef, GeometryRegionRole};
use serde::{Deserialize, Serialize};

/// Stable source identity of a reusable authored model.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredModelRef {
    pub module_path: String,
    pub model_id: String,
}

/// Stable source identity of a reusable authored material.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredMaterialRef {
    pub module_path: String,
    pub material_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredModelInputKind {
    Material,
    Region(GeometryRegionRole),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredModelInputRef {
    pub input_id: String,
    pub kind: AuthoredModelInputKind,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "source")]
pub enum AuthoredMaterialSource {
    Declaration(AuthoredMaterialRef),
    ModelInput(AuthoredModelInputRef),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "source")]
pub enum AuthoredPropertyRegionSource {
    Geometry(GeometryRegionRef),
    ModelInput(AuthoredModelInputRef),
}

/// Closed material-bearing formulation role. Boundary faces and raw topology are not variants.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "role", content = "region")]
pub enum AuthoredPropertyTarget {
    Beam(AuthoredPropertyRegionSource),
    Shell(AuthoredPropertyRegionSource),
    Solid(AuthoredPropertyRegionSource),
    Cell(AuthoredPropertyRegionSource),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredMaterialAssignment {
    pub material: AuthoredMaterialSource,
    pub target: AuthoredPropertyTarget,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredModelInclude {
    pub model: AuthoredModelRef,
}

/// Canonical reusable physical intent before a study binds inputs and a geometry revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredModel {
    pub module_path: String,
    pub model_id: String,
    pub property_inputs: Vec<AuthoredModelInputRef>,
    pub includes: Vec<AuthoredModelInclude>,
    pub material_assignments: Vec<AuthoredMaterialAssignment>,
}

impl AuthoredModel {
    pub fn validate(&self) -> Result<(), AuthoredModelError> {
        token("model module path", &self.module_path)?;
        token("model id", &self.model_id)?;
        let mut inputs = BTreeSet::new();
        for input in &self.property_inputs {
            input.validate()?;
            if !inputs.insert(input.input_id.as_str()) {
                return Err(AuthoredModelError::DuplicateInput(input.input_id.clone()));
            }
        }
        let mut includes = BTreeSet::new();
        for include in &self.includes {
            include.model.validate()?;
            if include.model.module_path == self.module_path
                && include.model.model_id == self.model_id
            {
                return Err(AuthoredModelError::SelfInclude);
            }
            if !includes.insert(&include.model) {
                return Err(AuthoredModelError::DuplicateInclude);
            }
        }
        let mut assignments = BTreeSet::new();
        for assignment in &self.material_assignments {
            assignment.validate(&inputs)?;
            if !assignments.insert(assignment) {
                return Err(AuthoredModelError::DuplicateMaterialAssignment);
            }
        }
        Ok(())
    }
}

impl AuthoredModelRef {
    fn validate(&self) -> Result<(), AuthoredModelError> {
        token("included model module path", &self.module_path)?;
        token("included model id", &self.model_id)
    }
}

impl AuthoredModelInputRef {
    fn validate(&self) -> Result<(), AuthoredModelError> {
        token("model input id", &self.input_id)
    }
}

impl AuthoredMaterialAssignment {
    fn validate(&self, inputs: &BTreeSet<&str>) -> Result<(), AuthoredModelError> {
        match &self.material {
            AuthoredMaterialSource::Declaration(material) => {
                token("material module path", &material.module_path)?;
                token("material id", &material.material_id)?;
            }
            AuthoredMaterialSource::ModelInput(input) => {
                input.validate()?;
                if input.kind != AuthoredModelInputKind::Material {
                    return Err(AuthoredModelError::WrongMaterialInputKind);
                }
                require_input(input, inputs)?;
            }
        }
        let (source, expected_role) = match &self.target {
            AuthoredPropertyTarget::Beam(source) => (source, GeometryRegionRole::Edge),
            AuthoredPropertyTarget::Shell(source) => (source, GeometryRegionRole::Face),
            AuthoredPropertyTarget::Solid(source) => (source, GeometryRegionRole::Solid),
            AuthoredPropertyTarget::Cell(source) => (source, GeometryRegionRole::Cell),
        };
        let actual_role = match source {
            AuthoredPropertyRegionSource::Geometry(region) => {
                token("region module path", &region.module_path)?;
                token("region geometry id", &region.geometry_id)?;
                token("region id", &region.region_id)?;
                region.role
            }
            AuthoredPropertyRegionSource::ModelInput(input) => {
                input.validate()?;
                require_input(input, inputs)?;
                let AuthoredModelInputKind::Region(role) = input.kind else {
                    return Err(AuthoredModelError::WrongRegionInputKind);
                };
                role
            }
        };
        if actual_role != expected_role {
            return Err(AuthoredModelError::WrongPropertyTargetRole {
                expected: expected_role,
                actual: actual_role,
            });
        }
        Ok(())
    }
}

fn require_input(
    input: &AuthoredModelInputRef,
    inputs: &BTreeSet<&str>,
) -> Result<(), AuthoredModelError> {
    if inputs.contains(input.input_id.as_str()) {
        Ok(())
    } else {
        Err(AuthoredModelError::UnknownInput(input.input_id.clone()))
    }
}

fn token(field: &'static str, value: &str) -> Result<(), AuthoredModelError> {
    if value.trim().is_empty() {
        Err(AuthoredModelError::InvalidToken(field))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthoredModelError {
    #[error("{0} must not be empty")]
    InvalidToken(&'static str),
    #[error("model input `{0}` is duplicated")]
    DuplicateInput(String),
    #[error("model cannot include itself")]
    SelfInclude,
    #[error("model include is duplicated")]
    DuplicateInclude,
    #[error("material assignment is duplicated")]
    DuplicateMaterialAssignment,
    #[error("material source input is not typed as material")]
    WrongMaterialInputKind,
    #[error("material target input is not typed as a region")]
    WrongRegionInputKind,
    #[error("model input `{0}` is not declared as a property input")]
    UnknownInput(String),
    #[error("material target role expected {expected:?} topology, found {actual:?}")]
    WrongPropertyTargetRole {
        expected: GeometryRegionRole,
        actual: GeometryRegionRole,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_assignment_requires_declared_strongly_typed_inputs() {
        let material = AuthoredModelInputRef {
            input_id: "material".into(),
            kind: AuthoredModelInputKind::Material,
        };
        let body = AuthoredModelInputRef {
            input_id: "body".into(),
            kind: AuthoredModelInputKind::Region(GeometryRegionRole::Solid),
        };
        AuthoredModel {
            module_path: "project/main.fea".into(),
            model_id: "bracket".into(),
            property_inputs: vec![material.clone(), body.clone()],
            includes: Vec::new(),
            material_assignments: vec![AuthoredMaterialAssignment {
                material: AuthoredMaterialSource::ModelInput(material),
                target: AuthoredPropertyTarget::Solid(AuthoredPropertyRegionSource::ModelInput(
                    body,
                )),
            }],
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn boundary_vertices_are_not_material_domains() {
        let model = AuthoredModel {
            module_path: "project/main.fea".into(),
            model_id: "invalid".into(),
            property_inputs: Vec::new(),
            includes: Vec::new(),
            material_assignments: vec![AuthoredMaterialAssignment {
                material: AuthoredMaterialSource::Declaration(AuthoredMaterialRef {
                    module_path: "project/main.fea".into(),
                    material_id: "steel".into(),
                }),
                target: AuthoredPropertyTarget::Solid(AuthoredPropertyRegionSource::Geometry(
                    GeometryRegionRef {
                        module_path: "project/main.fea".into(),
                        geometry_id: "part".into(),
                        region_id: "point".into(),
                        role: GeometryRegionRole::Vertex,
                    },
                )),
            }],
        };
        assert_eq!(
            model.validate(),
            Err(AuthoredModelError::WrongPropertyTargetRole {
                expected: GeometryRegionRole::Solid,
                actual: GeometryRegionRole::Vertex,
            })
        );
    }
}
