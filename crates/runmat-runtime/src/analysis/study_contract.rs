use runmat_analysis_core::{validate_model_against_geometry, CoordinateFrame};
use runmat_geometry_core::UnitSystem;

use super::{contracts, AnalysisSolverOptions, AnalysisStudySpec, ANALYSIS_VALIDATE_OP_VERSION};

pub(super) fn sanitize_study_sweep_id(sweep_id: &str) -> String {
    sweep_id
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

pub(super) fn validate_study_issue_codes(spec: &AnalysisStudySpec) -> Vec<String> {
    let mut issue_codes = Vec::new();

    if spec.study_id.trim().is_empty() {
        issue_codes.push("RM.FEA.STUDY.ID_EMPTY".to_string());
    }
    if spec.model.model_id.0.trim().is_empty() {
        issue_codes.push("RM.FEA.STUDY.MODEL_ID_EMPTY".to_string());
    }
    if spec.geometry.meshes.is_empty() {
        issue_codes.push("RM.FEA.STUDY.GEOMETRY_MESHES_EMPTY".to_string());
    }
    if spec.geometry.units == UnitSystem::Unspecified {
        issue_codes.push("RM.FEA.STUDY.GEOMETRY_UNITS_UNSPECIFIED".to_string());
    }
    if spec.execution.steps.is_empty() {
        issue_codes.push("RM.FEA.STUDY.EXECUTION_EMPTY".to_string());
    }
    if !immutable_value_inventory_is_exact(spec) {
        issue_codes.push("RM.FEA.STUDY.IMMUTABLE_VALUES_INVALID".to_string());
    }
    let model = &spec.model;
    if model.geometry_id != spec.geometry.geometry_id
        || model.geometry_revision != spec.geometry.revision
    {
        issue_codes.push("RM.FEA.STUDY.MODEL_GEOMETRY_MISMATCH".to_string());
    }
    if validate_model_against_geometry(model, spec.geometry.units, &CoordinateFrame::Global)
        .is_err()
    {
        issue_codes.push("RM.FEA.STUDY.MODEL_INVALID".to_string());
    }
    if spec.execution.validate_against(&model.sequence).is_err() {
        issue_codes.push("RM.FEA.STUDY.EXECUTION_PLAN_INVALID".to_string());
    }
    for execution in &spec.execution.steps {
        if let AnalysisSolverOptions::ElectromagneticPotential(options) = &execution.solver_options
        {
            if !options.residual_target.is_finite() || options.residual_target <= 0.0 {
                issue_codes
                    .push("RM.FEA.STUDY.ELECTROMAGNETIC_RESIDUAL_TARGET_INVALID".to_string());
            }
            if !options.harmonic_tolerance.is_finite() || options.harmonic_tolerance <= 0.0 {
                issue_codes
                    .push("RM.FEA.STUDY.ELECTROMAGNETIC_HARMONIC_TOLERANCE_INVALID".to_string());
            }
            if options.harmonic_max_iterations == 0 {
                issue_codes.push(
                    "RM.FEA.STUDY.ELECTROMAGNETIC_HARMONIC_MAX_ITERATIONS_INVALID".to_string(),
                );
            }
            if options.sweep_enabled
                && !options
                    .sweep_frequency_hz
                    .iter()
                    .all(|frequency_hz| frequency_hz.is_finite() && *frequency_hz > 0.0)
            {
                issue_codes
                    .push("RM.FEA.STUDY.ELECTROMAGNETIC_SWEEP_FREQUENCY_INVALID".to_string());
            }
        }
    }

    issue_codes
}

fn immutable_value_inventory_is_exact(spec: &AnalysisStudySpec) -> bool {
    contracts::validate_immutable_value_inventory(
        &spec.immutable_values,
        spec.model.signals.as_slice(),
    )
    .is_ok()
}

pub(super) fn study_issue_message(code: &str) -> &'static str {
    match code {
        "RM.FEA.STUDY.ID_EMPTY" => "study_id must be non-empty",
        "RM.FEA.STUDY.MODEL_ID_EMPTY" => "model.model_id must be non-empty",
        "RM.FEA.STUDY.GEOMETRY_MESHES_EMPTY" => "geometry must contain at least one mesh",
        "RM.FEA.STUDY.GEOMETRY_UNITS_UNSPECIFIED" => {
            "geometry.units must be specified (not unspecified)"
        }
        "RM.FEA.STUDY.EXECUTION_EMPTY" => "execution must contain at least one typed procedure step",
        "RM.FEA.STUDY.IMMUTABLE_VALUES_INVALID" => {
            "immutable values must be unique, normalized, content-bound, and referenced exactly by generator arguments"
        }
        "RM.FEA.STUDY.EXECUTION_PLAN_INVALID" => {
            "execution must map each canonical model procedure step to exactly one compatible solver"
        }
        "RM.FEA.STUDY.MODEL_GEOMETRY_MISMATCH" => {
            "study model geometry id or revision does not match the study geometry"
        }
        "RM.FEA.STUDY.MODEL_INVALID" => "study model failed FEA validation",
        "RM.FEA.STUDY.ELECTROMAGNETIC_RESIDUAL_TARGET_INVALID" => {
            "electromagnetic execution options.residual_target must be finite and positive"
        }
        "RM.FEA.STUDY.ELECTROMAGNETIC_HARMONIC_TOLERANCE_INVALID" => {
            "electromagnetic execution options.harmonic_tolerance must be finite and positive"
        }
        "RM.FEA.STUDY.ELECTROMAGNETIC_HARMONIC_MAX_ITERATIONS_INVALID" => {
            "electromagnetic execution options.harmonic_max_iterations must be greater than zero"
        }
        "RM.FEA.STUDY.ELECTROMAGNETIC_SWEEP_FREQUENCY_INVALID" => {
            "electromagnetic execution options.sweep_frequency_hz must contain finite positive values when sweep_enabled is true"
        }
        _ => "unrecognized study validation issue",
    }
}

/// Computes the authored study's placement-independent content identity.
///
/// The geometry source digest identifies the input bytes. Local source and generated artifact
/// paths identify storage locations, so they cannot participate in a portable study identity.
pub fn analysis_study_content_digest(spec: &AnalysisStudySpec) -> runmat_execution::Digest {
    let mut semantic_spec = spec.clone();
    semantic_spec.geometry.source.path.clear();
    semantic_spec.solver_mesh_artifact_path = None;
    semantic_spec.meshing_evidence_artifact_path = None;
    for value in &mut semantic_spec.immutable_values {
        value.relative_path.clear();
    }
    for execution in &mut semantic_spec.execution.steps {
        execution.solver_options.clear_solver_mesh_artifact_path();
    }
    let payload = serde_json::to_vec(&semantic_spec)
        .unwrap_or_else(|_| format!("{semantic_spec:?}").into_bytes());
    runmat_execution::Digest::sha256(&payload)
}

pub(super) fn study_fingerprint(spec: &AnalysisStudySpec) -> String {
    study_fingerprint_from_digest(analysis_study_content_digest(spec))
}

pub(super) fn study_fingerprint_from_digest(digest: runmat_execution::Digest) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in digest.bytes() {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing to a string cannot fail");
    }
    format!("sha256:{encoded}")
}

pub(super) fn study_operation_sequence(
    _spec: &AnalysisStudySpec,
    run_op_versions: &[String],
) -> Vec<String> {
    let mut operation_sequence = Vec::with_capacity(1 + run_op_versions.len());
    operation_sequence.push(ANALYSIS_VALIDATE_OP_VERSION.to_string());
    operation_sequence.extend(run_op_versions.iter().cloned());
    operation_sequence
}
