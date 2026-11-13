use thiserror::Error;

use crate::{
    problem::{coordinate_frame::CoordinateFrame, model::AnalysisModel},
    ValidatedAnalysisSequence,
};

#[derive(Debug, Clone, PartialEq, Error)]
pub enum AnalysisValidationError {
    #[error(
        "ANALYSIS_VALIDATION_MISSING_MATERIALS: analysis model must include at least one material"
    )]
    MissingMaterials,
    #[error("ANALYSIS_VALIDATION_MISSING_CONDITIONS: analysis model must include at least one constraint or natural boundary")]
    MissingConstraintsOrBoundaries,
    #[error("ANALYSIS_VALIDATION_INVALID_ID: {kind} id must be non-empty")]
    InvalidId { kind: &'static str },
    #[error("ANALYSIS_VALIDATION_DUPLICATE_ID: duplicate {kind} id `{id}`")]
    DuplicateId { kind: &'static str, id: String },
    #[error("ANALYSIS_VALIDATION_INVALID_TARGET: {owner_id} has an invalid target: {detail}")]
    InvalidTarget { owner_id: String, detail: String },
    #[error(
        "ANALYSIS_VALIDATION_INVALID_FRAME: coordinate frame is not finite and non-degenerate"
    )]
    InvalidFrame,
    #[error("ANALYSIS_VALIDATION_INVALID_MATERIAL: material `{material_id}` is invalid: {detail}")]
    InvalidMaterial { material_id: String, detail: String },
    #[error(
        "ANALYSIS_VALIDATION_INVALID_MATERIAL_ASSIGNMENT: material assignment for region `{region_id}` is invalid: {detail}"
    )]
    InvalidMaterialAssignment { region_id: String, detail: String },
    #[error(
        "ANALYSIS_VALIDATION_INVALID_CONSTRAINT: constraint `{constraint_id}` is invalid: {detail}"
    )]
    InvalidConstraint {
        constraint_id: String,
        detail: String,
    },
    #[error("ANALYSIS_VALIDATION_INVALID_BOUNDARY: boundary `{boundary_id}` is invalid: {detail}")]
    InvalidBoundary { boundary_id: String, detail: String },
    #[error(
        "ANALYSIS_VALIDATION_INVALID_CONNECTOR: connector `{connector_id}` is invalid: {detail}"
    )]
    InvalidConnector {
        connector_id: String,
        detail: String,
    },
    #[error(
        "ANALYSIS_VALIDATION_INVALID_INTERFACE: interface `{interface_id}` is invalid: {detail}"
    )]
    InvalidInterface {
        interface_id: String,
        detail: String,
    },
    #[error("ANALYSIS_VALIDATION_INVALID_SIGNAL: signal `{signal_id}` is invalid: {detail}")]
    InvalidSignal { signal_id: String, detail: String },
    #[error("ANALYSIS_VALIDATION_INVALID_DOMAIN: {domain} domain is invalid: {detail}")]
    InvalidDomain {
        domain: &'static str,
        detail: String,
    },
    #[error(
        "ANALYSIS_VALIDATION_INVALID_EXCITATION: excitation `{excitation_id}` is invalid: {detail}"
    )]
    InvalidExcitation {
        excitation_id: String,
        detail: String,
    },
    #[error(
        "ANALYSIS_VALIDATION_INVALID_EXCITATION_CASE: excitation case `{case_id}` is invalid: {detail}"
    )]
    InvalidExcitationCase { case_id: String, detail: String },
    #[error(
        "ANALYSIS_VALIDATION_INVALID_INITIAL_CONDITION: initial condition `{initial_condition_id}` is invalid: {detail}"
    )]
    InvalidInitialCondition {
        initial_condition_id: String,
        detail: String,
    },
    #[error(
        "ANALYSIS_VALIDATION_INVALID_RESPONSE_SPECTRUM: response spectrum `{spectrum_id}` is invalid: {detail}"
    )]
    InvalidResponseSpectrum { spectrum_id: String, detail: String },
    #[error(
        "ANALYSIS_VALIDATION_INVALID_RANDOM_EXCITATION: random excitation `{excitation_id}` is invalid: {detail}"
    )]
    InvalidRandomExcitation {
        excitation_id: String,
        detail: String,
    },
    #[error("ANALYSIS_VALIDATION_INVALID_SEQUENCE: sequence `{sequence_id}` is invalid: {detail}")]
    InvalidSequence { sequence_id: String, detail: String },
    #[error("ANALYSIS_VALIDATION_INVALID_PROCEDURE_STEP: procedure step `{step_id}` is invalid: {detail}")]
    InvalidProcedureStep { step_id: String, detail: String },
    #[error("ANALYSIS_VALIDATION_INVALID_STATE_TRANSITION: state transition `{transition_id}` is invalid: {detail}")]
    InvalidStateTransition {
        transition_id: String,
        detail: String,
    },
    #[error("ANALYSIS_VALIDATION_INVALID_OBSERVATION: observation `{observation_id}` is invalid: {detail}")]
    InvalidObservation {
        observation_id: String,
        detail: String,
    },
    #[error("ANALYSIS_VALIDATION_INVALID_MONITOR: monitor `{monitor_id}` is invalid: {detail}")]
    InvalidMonitor { monitor_id: String, detail: String },
    #[error(
        "ANALYSIS_VALIDATION_INVALID_ACCURACY_PLAN: accuracy plan `{plan_id}` is invalid: {detail}"
    )]
    InvalidAccuracyPlan { plan_id: String, detail: String },
    #[error("ANALYSIS_DEPENDENCY_GRAPH_RESOURCE_LIMIT: requested {requested_bytes} bytes exceeds {maximum_bytes}")]
    DependencyGraphResourceLimit {
        requested_bytes: u64,
        maximum_bytes: u64,
    },
    #[error("ANALYSIS_DEPENDENCY_GRAPH_RESOURCE_OVERFLOW: dependency index inventory overflows")]
    DependencyGraphResourceOverflow,
    #[error(
        "ANALYSIS_VALIDATION_FRAME_MISMATCH: model frame {model:?} does not match geometry frame {geometry:?}"
    )]
    FrameMismatch {
        model: Box<CoordinateFrame>,
        geometry: Box<CoordinateFrame>,
    },
}

pub fn validate_model(model: &AnalysisModel) -> Result<(), AnalysisValidationError> {
    validated_model_graph(model).map(drop)
}

/// Validates solver-independent procedure, dependency, and state-transfer semantics before an
/// authored model has been realized. Model-object existence and initial-condition coverage are
/// deliberately checked later by [`validate_model_sequence`].
pub fn validate_analysis_sequence(
    sequence: &crate::AnalysisSequence,
) -> Result<ValidatedAnalysisSequence, AnalysisValidationError> {
    crate::validate_sequence::validate_sequence_contract(sequence)
}

/// Reports whether a physics family can consume a canonical state kind.
///
/// Authoring frontends use this before constructing state inputs; complete model validation uses
/// the same authority again after realization.
pub fn physics_family_can_consume_state(
    family: crate::PhysicsFamily,
    kind: crate::StateKind,
) -> bool {
    crate::validate_sequence::family_can_consume_state(family, kind)
}

/// Validates reusable constitutive declarations before they are assigned to a realized model.
pub fn validate_material_definitions(
    materials: &[crate::MaterialModel],
) -> Result<(), AnalysisValidationError> {
    crate::validate_materials::validate_material_definitions(materials)
}

/// Validates the complete authored model and returns its immutable semantic
/// sequence topology for execution planning.
pub fn validate_model_sequence(
    model: &AnalysisModel,
) -> Result<ValidatedAnalysisSequence, AnalysisValidationError> {
    let graph = validated_model_graph(model)?;
    Ok(crate::validate_sequence::graph::owned_topology(
        &model.sequence,
        &graph,
    ))
}

// Both public boundaries check the original model through this one semantic authority.
// Validation-only callers do not materialize or clone an execution-planning sequence.
fn validated_model_graph(
    model: &AnalysisModel,
) -> Result<crate::validate_sequence::graph::ValidatedGraph<'_>, AnalysisValidationError> {
    if model.materials.is_empty() {
        return Err(AnalysisValidationError::MissingMaterials);
    }
    if model.constraints.is_empty() && model.boundaries.is_empty() {
        return Err(AnalysisValidationError::MissingConstraintsOrBoundaries);
    }
    crate::validate_authored::validate_authored_model(model)?;
    crate::validate_sequence::validate_sequence(model)
}

pub fn validate_model_against_geometry(
    model: &AnalysisModel,
    geometry_frame: &CoordinateFrame,
) -> Result<(), AnalysisValidationError> {
    validate_model(model)?;

    if &model.frame != geometry_frame {
        return Err(AnalysisValidationError::FrameMismatch {
            model: Box::new(model.frame.clone()),
            geometry: Box::new(geometry_frame.clone()),
        });
    }
    Ok(())
}

/// Additional flat dependency-index construction and ancestry-query workspace.
/// Excludes borrowed model storage, other semantic validators, error strings and owned planning.
pub fn dependency_graph_resident_bytes_upper_bound(
    sequence: &crate::AnalysisSequence,
) -> Result<u64, AnalysisValidationError> {
    crate::validate_sequence::graph::resident_bytes(sequence)
}
/// Validate the original model after admitting its dependency-index workspace.
/// This is a component budget, not full model-validation or whole-task resource admission.
pub fn validate_model_with_dependency_graph_budget(
    model: &AnalysisModel,
    maximum_bytes: u64,
) -> Result<(), AnalysisValidationError> {
    let requested_bytes = dependency_graph_resident_bytes_upper_bound(&model.sequence)?;
    if requested_bytes > maximum_bytes {
        return Err(AnalysisValidationError::DependencyGraphResourceLimit {
            requested_bytes,
            maximum_bytes,
        });
    }
    validate_model(model)
}
