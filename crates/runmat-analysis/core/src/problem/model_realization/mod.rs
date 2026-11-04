//! Conversion from bound authoring intent to the topology-independent solver model.

use std::collections::BTreeMap;

use runmat_geometry_core::ResolvedGeometryRegions;

use crate::{AnalysisModel, AnalysisModelId, AuthoredMaterialRef, MaterialModel, RealizedStudy};

mod constraints;
mod domains;
mod physics;
mod properties;
mod targets;

pub use properties::{
    RealizedLayupDefinition, RealizedModelProperties, RealizedOrientationDefinition,
    RealizedPropertyAssignment, RealizedSectionDefinition,
};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RealizedAnalysisModel {
    pub model: AnalysisModel,
    pub properties: RealizedModelProperties,
}

/// Inputs whose identity is fixed only once a study geometry has been materialized.
pub struct AnalysisModelRealizationInput<'a> {
    pub study: &'a RealizedStudy,
    pub geometry_id: &'a str,
    pub geometry_revision: u32,
    pub units:
        runmat_types::physical::CoordinateUnits<runmat_types::physical::CanonicalCoordinates>,
    pub regions: &'a ResolvedGeometryRegions,
    pub materials: &'a BTreeMap<AuthoredMaterialRef, MaterialModel>,
}

/// Builds the canonical solver-facing model with authored regions bound to exact geometry entities.
///
/// Formulation properties are compiled against solver topology after a mesh exists.
pub fn realize_analysis_model(
    input: AnalysisModelRealizationInput<'_>,
) -> Result<RealizedAnalysisModel, AnalysisModelRealizationError> {
    let AnalysisModelRealizationInput {
        study,
        geometry_id,
        geometry_revision,
        units,
        regions,
        materials,
    } = input;
    let models = study
        .included_models
        .iter()
        .chain(std::iter::once(&study.model))
        .collect::<Vec<_>>();

    for model in &models {
        model
            .validate()
            .map_err(|error| AnalysisModelRealizationError::InvalidModel(error.to_string()))?;
        if !model.inputs.is_empty() || !model.parameterized_signals.is_empty() {
            return Err(AnalysisModelRealizationError::UnboundModel(
                model.model_id.clone(),
            ));
        }
    }

    let target_resolver = targets::TargetResolver::new(&study.geometry.definition, regions);
    let material_resolution =
        targets::MaterialResolution::new(materials, &models, &target_resolver)?;
    let properties = properties::realize_model_properties(&models, &target_resolver)?;
    let (thermal, thermo_mechanical, electromagnetic) = domains::realize(&models)?;
    let mut sequence = study.sequence.clone();
    for step in &mut sequence.steps {
        for observation in &mut step.observations {
            let Some(spec) = observation.spec_mut() else {
                continue;
            };
            match &mut spec.target {
                crate::ObservationTarget::Spatial(crate::ExcitationTarget::Region(target)) => {
                    target_resolver.bind_existing_region_target(target)?;
                }
                crate::ObservationTarget::RegionPair { first, second } => {
                    target_resolver.bind_existing_region_target(first)?;
                    target_resolver.bind_existing_region_target(second)?;
                }
                _ => {}
            }
        }
    }
    let mut model = AnalysisModel {
        model_id: AnalysisModelId(study.model.model_id.clone()),
        geometry_id: geometry_id.to_string(),
        geometry_revision,
        units,
        frame: crate::CoordinateFrame::Global,
        materials: material_resolution.materials,
        material_assignments: material_resolution.assignments,
        structural: None,
        thermal,
        thermo_mechanical,
        electro_thermal: None,
        electromagnetic,
        cfd: None,
        interfaces: Vec::new(),
        constraints: Vec::new(),
        boundaries: Vec::new(),
        connectors: Vec::new(),
        signals: Vec::new(),
        excitations: Vec::new(),
        excitation_cases: Vec::new(),
        initial_conditions: Vec::new(),
        response_spectra: Vec::new(),
        random_excitations: Vec::new(),
        sequence,
    };
    for authored in models {
        constraints::extend_constraints(
            &mut model.constraints,
            &authored.constraints,
            &target_resolver,
        )?;
        physics::extend_model_physics(&mut model, authored, &target_resolver)?;
    }
    Ok(RealizedAnalysisModel { model, properties })
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AnalysisModelRealizationError {
    #[error("included/root models contain conflicting `{0}` domain declarations")]
    ConflictingDomain(&'static str),
    #[error("authored model is invalid: {0}")]
    InvalidModel(String),
    #[error("model `{0}` still contains unbound inputs")]
    UnboundModel(String),
    #[error("material `{module_path}::{material_id}` is unavailable")]
    MissingMaterial {
        module_path: String,
        material_id: String,
    },
    #[error("authored region target is not fully bound")]
    UnboundRegion,
    #[error("geometry materialization did not publish authored region `{0}`")]
    MissingResolvedRegion(String),
    #[error("geometry region resolution is incompatible with its authored target: {0}")]
    InvalidRegionResolution(String),
    #[error(
        "authored target refers to geometry `{module_path}::{geometry_id}`, not the study geometry"
    )]
    RegionFromWrongGeometry {
        module_path: String,
        geometry_id: String,
    },
    #[error("initial condition `{0}` must be activated by exactly one procedure step")]
    InitialConditionActivation(String),
}
