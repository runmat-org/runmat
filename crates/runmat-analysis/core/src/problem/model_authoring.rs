use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{GeometryRegionRef, GeometryRegionRole};
use serde::{Deserialize, Serialize};

use super::authored_properties::{
    AuthoredLayupDefinition, AuthoredLayupSource, AuthoredOrientationDefinition,
    AuthoredOrientationSource, AuthoredPropertyAssignment, AuthoredSectionDefinition,
    AuthoredSectionSource,
};

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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredModel {
    pub module_path: String,
    pub model_id: String,
    pub property_inputs: Vec<AuthoredModelInputRef>,
    pub includes: Vec<AuthoredModelInclude>,
    pub sections: Vec<AuthoredSectionDefinition>,
    pub layups: Vec<AuthoredLayupDefinition>,
    pub orientations: Vec<AuthoredOrientationDefinition>,
    pub material_assignments: Vec<AuthoredMaterialAssignment>,
    pub property_assignments: Vec<AuthoredPropertyAssignment>,
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
        let mut sections = BTreeMap::new();
        for section in &self.sections {
            section
                .validate()
                .map_err(|error| AuthoredModelError::InvalidProperty(error.to_string()))?;
            if sections.insert(section.id(), section.is_beam()).is_some() {
                return Err(AuthoredModelError::DuplicateSection(
                    section.id().to_string(),
                ));
            }
        }
        let mut orientations = BTreeSet::new();
        for orientation in &self.orientations {
            orientation
                .validate()
                .map_err(|error| AuthoredModelError::InvalidProperty(error.to_string()))?;
            if !orientations.insert(orientation.id()) {
                return Err(AuthoredModelError::DuplicateOrientation(
                    orientation.id().to_string(),
                ));
            }
        }
        let mut layups = BTreeSet::new();
        for layup in &self.layups {
            layup
                .validate()
                .map_err(|error| AuthoredModelError::InvalidProperty(error.to_string()))?;
            if !layups.insert(layup.layup_id.as_str()) {
                return Err(AuthoredModelError::DuplicateLayup(layup.layup_id.clone()));
            }
            validate_local_property_ref(
                &layup.reference_orientation,
                &self.module_path,
                &self.model_id,
            )?;
            if !orientations.contains(layup.reference_orientation.property_id.as_str()) {
                return Err(AuthoredModelError::UnknownOrientation(
                    layup.reference_orientation.property_id.clone(),
                ));
            }
            for ply in &layup.plies {
                validate_material_source(&ply.material, &inputs)?;
            }
        }
        let mut properties = BTreeSet::new();
        let mut section_targets = BTreeSet::new();
        let mut layup_targets = BTreeSet::new();
        let mut orientation_targets = BTreeSet::new();
        for assignment in &self.property_assignments {
            validate_property_assignment(
                assignment,
                &self.module_path,
                &self.model_id,
                &sections,
                &layups,
                &orientations,
            )?;
            if !properties.insert(assignment) {
                return Err(AuthoredModelError::DuplicatePropertyAssignment);
            }
            match assignment {
                AuthoredPropertyAssignment::Section { target, .. } => {
                    if !section_targets.insert(target) {
                        return Err(AuthoredModelError::ConflictingSectionAssignment);
                    }
                }
                AuthoredPropertyAssignment::Layup { target, .. } => {
                    if !layup_targets.insert(target) {
                        return Err(AuthoredModelError::ConflictingLayupAssignment);
                    }
                }
                AuthoredPropertyAssignment::Orientation { target, .. } => {
                    if !orientation_targets.insert(target) {
                        return Err(AuthoredModelError::ConflictingOrientationAssignment);
                    }
                }
            }
        }
        for material in &self.material_assignments {
            if let AuthoredPropertyTarget::Shell(source) = &material.target {
                if layup_targets.contains(source) {
                    return Err(AuthoredModelError::CompositeAndHomogeneousOwnership);
                }
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
        validate_material_source(&self.material, inputs)?;
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

fn validate_material_source(
    source: &AuthoredMaterialSource,
    inputs: &BTreeSet<&str>,
) -> Result<(), AuthoredModelError> {
    match source {
        AuthoredMaterialSource::Declaration(material) => {
            token("material module path", &material.module_path)?;
            token("material id", &material.material_id)
        }
        AuthoredMaterialSource::ModelInput(input) => {
            input.validate()?;
            if input.kind != AuthoredModelInputKind::Material {
                return Err(AuthoredModelError::WrongMaterialInputKind);
            }
            require_input(input, inputs)
        }
    }
}

fn validate_property_assignment(
    assignment: &AuthoredPropertyAssignment,
    module_path: &str,
    model_id: &str,
    sections: &BTreeMap<&str, bool>,
    layups: &BTreeSet<&str>,
    orientations: &BTreeSet<&str>,
) -> Result<(), AuthoredModelError> {
    match assignment {
        AuthoredPropertyAssignment::Section { section, target } => {
            let AuthoredSectionSource::Definition(reference) = section;
            validate_local_property_ref(reference, module_path, model_id)?;
            let Some(is_beam) = sections.get(reference.property_id.as_str()) else {
                return Err(AuthoredModelError::UnknownSection(
                    reference.property_id.clone(),
                ));
            };
            if (*is_beam && !matches!(target, AuthoredPropertyTarget::Beam(_)))
                || (!*is_beam && !matches!(target, AuthoredPropertyTarget::Shell(_)))
            {
                return Err(AuthoredModelError::WrongSectionTarget);
            }
        }
        AuthoredPropertyAssignment::Layup { layup, target } => {
            let AuthoredLayupSource::Definition(reference) = layup;
            validate_local_property_ref(reference, module_path, model_id)?;
            if !layups.contains(reference.property_id.as_str()) {
                return Err(AuthoredModelError::UnknownLayup(
                    reference.property_id.clone(),
                ));
            }
            if property_source_role(target) != GeometryRegionRole::Face {
                return Err(AuthoredModelError::WrongLayupTarget);
            }
        }
        AuthoredPropertyAssignment::Orientation {
            orientation,
            target,
        } => {
            let AuthoredOrientationSource::Definition(reference) = orientation;
            validate_local_property_ref(reference, module_path, model_id)?;
            if !orientations.contains(reference.property_id.as_str()) {
                return Err(AuthoredModelError::UnknownOrientation(
                    reference.property_id.clone(),
                ));
            }
            if matches!(target, AuthoredPropertyTarget::Cell(_)) {
                return Err(AuthoredModelError::WrongOrientationTarget);
            }
        }
    }
    Ok(())
}

fn validate_local_property_ref(
    reference: &super::authored_properties::AuthoredPropertyDefinitionRef,
    module_path: &str,
    model_id: &str,
) -> Result<(), AuthoredModelError> {
    reference
        .validate()
        .map_err(|error| AuthoredModelError::InvalidProperty(error.to_string()))?;
    if reference.module_path != module_path || reference.model_id != model_id {
        return Err(AuthoredModelError::ForeignPropertyReference);
    }
    Ok(())
}

fn property_source_role(source: &AuthoredPropertyRegionSource) -> GeometryRegionRole {
    match source {
        AuthoredPropertyRegionSource::Geometry(region) => region.role,
        AuthoredPropertyRegionSource::ModelInput(input) => match input.kind {
            AuthoredModelInputKind::Region(role) => role,
            AuthoredModelInputKind::Material => GeometryRegionRole::Vertex,
        },
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
    #[error("section `{0}` is duplicated")]
    DuplicateSection(String),
    #[error("layup `{0}` is duplicated")]
    DuplicateLayup(String),
    #[error("orientation `{0}` is duplicated")]
    DuplicateOrientation(String),
    #[error("property definition is invalid: {0}")]
    InvalidProperty(String),
    #[error("property assignment is duplicated")]
    DuplicatePropertyAssignment,
    #[error("multiple section assignments control the same target")]
    ConflictingSectionAssignment,
    #[error("multiple layup assignments control the same target")]
    ConflictingLayupAssignment,
    #[error("multiple orientation assignments control the same target")]
    ConflictingOrientationAssignment,
    #[error("a shell target cannot have both homogeneous material and laminate ownership")]
    CompositeAndHomogeneousOwnership,
    #[error("unknown section `{0}`")]
    UnknownSection(String),
    #[error("unknown layup `{0}`")]
    UnknownLayup(String),
    #[error("unknown orientation `{0}`")]
    UnknownOrientation(String),
    #[error("model-local property reference has a different module or model owner")]
    ForeignPropertyReference,
    #[error("section assignments require beam or shell targets")]
    WrongSectionTarget,
    #[error("layup assignments require shell-face targets")]
    WrongLayupTarget,
    #[error("orientation assignments do not support cell targets")]
    WrongOrientationTarget,
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
            sections: Vec::new(),
            layups: Vec::new(),
            orientations: Vec::new(),
            material_assignments: vec![AuthoredMaterialAssignment {
                material: AuthoredMaterialSource::ModelInput(material),
                target: AuthoredPropertyTarget::Solid(AuthoredPropertyRegionSource::ModelInput(
                    body,
                )),
            }],
            property_assignments: Vec::new(),
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
            sections: Vec::new(),
            layups: Vec::new(),
            orientations: Vec::new(),
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
            property_assignments: Vec::new(),
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
