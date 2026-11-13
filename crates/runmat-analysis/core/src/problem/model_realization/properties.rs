use crate::{
    AuthoredLayupDefinition, AuthoredLayupSource, AuthoredModel, AuthoredOrientationDefinition,
    AuthoredOrientationSource, AuthoredPropertyAssignment, AuthoredPropertyDefinitionRef,
    AuthoredSectionDefinition, AuthoredSectionSource, PropertyRegionTarget,
};

use super::{targets::TargetResolver, AnalysisModelRealizationError};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct RealizedModelProperties {
    pub sections: Vec<RealizedSectionDefinition>,
    pub layups: Vec<RealizedLayupDefinition>,
    pub orientations: Vec<RealizedOrientationDefinition>,
    pub assignments: Vec<RealizedPropertyAssignment>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RealizedSectionDefinition {
    pub definition_id: AuthoredPropertyDefinitionRef,
    pub definition: AuthoredSectionDefinition,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RealizedLayupDefinition {
    pub definition_id: AuthoredPropertyDefinitionRef,
    pub definition: AuthoredLayupDefinition,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RealizedOrientationDefinition {
    pub definition_id: AuthoredPropertyDefinitionRef,
    pub definition: AuthoredOrientationDefinition,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", tag = "assignment")]
pub enum RealizedPropertyAssignment {
    Section {
        section: AuthoredPropertyDefinitionRef,
        target: PropertyRegionTarget,
    },
    Layup {
        layup: AuthoredPropertyDefinitionRef,
        target: PropertyRegionTarget,
    },
    Orientation {
        orientation: AuthoredPropertyDefinitionRef,
        target: PropertyRegionTarget,
    },
}

pub(super) fn realize_model_properties(
    models: &[&AuthoredModel],
    targets: &TargetResolver<'_>,
) -> Result<RealizedModelProperties, AnalysisModelRealizationError> {
    let mut realized = RealizedModelProperties::default();
    for model in models {
        realized
            .sections
            .extend(
                model
                    .sections
                    .iter()
                    .cloned()
                    .map(|definition| RealizedSectionDefinition {
                        definition_id: definition_id(model, definition.id()),
                        definition,
                    }),
            );
        realized
            .layups
            .extend(
                model
                    .layups
                    .iter()
                    .cloned()
                    .map(|definition| RealizedLayupDefinition {
                        definition_id: definition_id(model, &definition.layup_id),
                        definition,
                    }),
            );
        realized
            .orientations
            .extend(model.orientations.iter().cloned().map(|definition| {
                RealizedOrientationDefinition {
                    definition_id: definition_id(model, definition.id()),
                    definition,
                }
            }));
        for assignment in &model.property_assignments {
            realized.assignments.push(match assignment {
                AuthoredPropertyAssignment::Section { section, target } => {
                    let AuthoredSectionSource::Definition(section) = section;
                    RealizedPropertyAssignment::Section {
                        section: section.clone(),
                        target: targets.property_target(target)?,
                    }
                }
                AuthoredPropertyAssignment::Layup { layup, target } => {
                    let AuthoredLayupSource::Definition(layup) = layup;
                    RealizedPropertyAssignment::Layup {
                        layup: layup.clone(),
                        target: targets.shell_property_target(target)?,
                    }
                }
                AuthoredPropertyAssignment::Orientation {
                    orientation,
                    target,
                } => {
                    let AuthoredOrientationSource::Definition(orientation) = orientation;
                    RealizedPropertyAssignment::Orientation {
                        orientation: orientation.clone(),
                        target: targets.property_target(target)?,
                    }
                }
            });
        }
    }
    Ok(realized)
}

fn definition_id(model: &AuthoredModel, property_id: &str) -> AuthoredPropertyDefinitionRef {
    AuthoredPropertyDefinitionRef {
        module_path: model.module_path.clone(),
        model_id: model.model_id.clone(),
        property_id: property_id.to_owned(),
    }
}
