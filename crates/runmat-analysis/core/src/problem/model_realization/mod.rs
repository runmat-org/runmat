//! Conversion from bound authoring intent to the topology-independent solver model.

use std::collections::BTreeMap;

use runmat_geometry_core::UnitSystem;

use crate::{AnalysisModel, AnalysisModelId, AuthoredMaterialRef, MaterialModel, RealizedStudy};

mod constraints;
mod physics;
mod targets;

/// Inputs whose identity is fixed only once a study geometry has been materialized.
pub struct AnalysisModelRealizationInput<'a> {
    pub study: &'a RealizedStudy,
    pub geometry_id: &'a str,
    pub geometry_revision: u32,
    pub units: UnitSystem,
    pub materials: &'a BTreeMap<AuthoredMaterialRef, MaterialModel>,
}

/// Builds the canonical solver-facing model without resolving mesh entities.
///
/// Region ids remain semantic selections here. Exact entities and formulation properties are
/// compiled after a solver mesh exists.
pub fn realize_analysis_model(
    input: AnalysisModelRealizationInput<'_>,
) -> Result<AnalysisModel, AnalysisModelRealizationError> {
    let AnalysisModelRealizationInput {
        study,
        geometry_id,
        geometry_revision,
        units,
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
        if !model.sections.is_empty()
            || !model.property_assignments.is_empty()
            || !model.layups.is_empty()
            || !model.orientations.is_empty()
        {
            return Err(
                AnalysisModelRealizationError::UnsupportedPropertyRealization(
                    model.model_id.clone(),
                ),
            );
        }
    }

    let target_resolver = targets::TargetResolver::new(&study.geometry.definition);
    let material_resolution =
        targets::MaterialResolution::new(materials, &models, &target_resolver)?;
    let mut model = AnalysisModel {
        model_id: AnalysisModelId(study.model.model_id.clone()),
        geometry_id: geometry_id.to_string(),
        geometry_revision,
        units,
        frame: crate::CoordinateFrame::Global,
        materials: material_resolution.materials,
        material_assignments: material_resolution.assignments,
        structural: None,
        thermal: None,
        thermo_mechanical: None,
        electro_thermal: None,
        electromagnetic: None,
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
        sequence: study.sequence.clone(),
    };
    for authored in models {
        constraints::extend_constraints(
            &mut model.constraints,
            &authored.constraints,
            &target_resolver,
        )?;
        physics::extend_model_physics(&mut model, authored, &target_resolver)?;
    }
    Ok(model)
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AnalysisModelRealizationError {
    #[error("authored model is invalid: {0}")]
    InvalidModel(String),
    #[error("model `{0}` still contains unbound inputs")]
    UnboundModel(String),
    #[error("model `{0}` uses layup, orientation, or section assignments that require the mesh property compiler")]
    UnsupportedPropertyRealization(String),
    #[error("material `{module_path}::{material_id}` is unavailable")]
    MissingMaterial {
        module_path: String,
        material_id: String,
    },
    #[error("authored region target is not fully bound")]
    UnboundRegion,
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
