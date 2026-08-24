use std::collections::BTreeMap;

use runmat_analysis_core::{EvidenceConfidence, LoadKind, MaterialAssignment};
use runmat_meshing_core::MeshingRequestSettings;

use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};

use super::{
    analysis_create_model_op, persist_study_evidence, profile_supports_run_kind, study_fingerprint,
    validate_study_issue_codes, AnalysisCreateModelIntentSpec, AnalysisStudyAuthoringData,
    AnalysisStudyAuthoringEvidence, AnalysisStudyAuthoringIntent, AnalysisStudyDiagramObservation,
    AnalysisStudySpec, ANALYSIS_AUTHOR_STUDY_OPERATION, ANALYSIS_AUTHOR_STUDY_OP_VERSION,
};

pub fn analysis_author_study_op(
    intent: AnalysisStudyAuthoringIntent,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisStudyAuthoringData>, OperationErrorEnvelope> {
    if intent.study_id.trim().is_empty() {
        return Err(author_study_error(
            &context,
            "RM.FEA.AUTHOR_STUDY.ID_EMPTY",
            "study_id must be non-empty",
            BTreeMap::new(),
        ));
    }
    if let Err(message) = intent.mesh_summary.validate() {
        return Err(author_study_error(
            &context,
            "RM.FEA.AUTHOR_STUDY.MESH_SUMMARY_INVALID",
            message,
            BTreeMap::from([("mesh_id".to_string(), intent.mesh_summary.mesh_id.clone())]),
        ));
    }
    if !intent.mesh_summary.solve_ready {
        return Err(author_study_error(
            &context,
            "RM.FEA.AUTHOR_STUDY.MESH_NOT_SOLVE_READY",
            "analysis mesh summary is not solve-ready",
            BTreeMap::from([
                ("mesh_id".to_string(), intent.mesh_summary.mesh_id.clone()),
                (
                    "validation_error_code".to_string(),
                    intent
                        .mesh_summary
                        .validation_error_code
                        .clone()
                        .unwrap_or_default(),
                ),
            ]),
        ));
    }
    if !profile_supports_run_kind(intent.profile, intent.run_kind) {
        return Err(author_study_error(
            &context,
            "RM.FEA.AUTHOR_STUDY.RUN_KIND_PROFILE_MISMATCH",
            "profile does not support requested run kind",
            BTreeMap::from([
                ("profile".to_string(), format!("{:?}", intent.profile)),
                ("run_kind".to_string(), format!("{:?}", intent.run_kind)),
            ]),
        ));
    }

    let (physical_region_id, physical_region_source) = select_authoring_physical_region(
        &intent.mesh_summary,
        intent.physical_region_id.as_deref(),
        intent.diagram_observation.as_ref(),
    )
    .map_err(|message| {
        author_study_error(
            &context,
            "RM.FEA.AUTHOR_STUDY.PHYSICAL_REGION_UNAVAILABLE",
            message,
            BTreeMap::from([("mesh_id".to_string(), intent.mesh_summary.mesh_id.clone())]),
        )
    })?;
    let model_id = intent
        .model_id
        .clone()
        .filter(|model_id| !model_id.trim().is_empty())
        .unwrap_or_else(|| format!("{}_model", intent.study_id.trim()));
    let create_model_intent = AnalysisCreateModelIntentSpec {
        model_id: model_id.clone(),
        profile: intent.profile,
    };
    let mut model = analysis_create_model_op(
        &intent.geometry,
        create_model_intent.clone(),
        context.clone(),
    )?
    .data;
    let material_id = model
        .materials
        .first()
        .map(|material| material.material_id.clone())
        .unwrap_or_else(|| "mat_default_steel".to_string());
    model.material_assignments = vec![MaterialAssignment {
        region_id: physical_region_id.clone(),
        expected_material_id: material_id.clone(),
        assigned_material_id: material_id,
        confidence: EvidenceConfidence::Inferred,
    }];
    let (boundary_condition_region_id, boundary_condition_region_source) =
        if model.boundary_conditions.is_empty() {
            (None, None)
        } else {
            let (region_id, source) = select_authoring_boundary_region(
                &intent.mesh_summary,
                intent.boundary_condition_region_id.as_deref(),
                intent
                    .diagram_observation
                    .as_ref()
                    .and_then(|observation| observation.boundary_condition_region_id.as_deref()),
                None,
            )
            .map_err(|message| {
                author_study_error(
                    &context,
                    "RM.FEA.AUTHOR_STUDY.BOUNDARY_CONDITION_REGION_UNAVAILABLE",
                    message,
                    BTreeMap::from([("mesh_id".to_string(), intent.mesh_summary.mesh_id.clone())]),
                )
            })?;
            (Some(region_id), Some(source))
        };
    let (driving_condition_region_id, driving_condition_region_source) = if model.loads.is_empty() {
        (None, None)
    } else {
        let (region_id, source) = select_authoring_boundary_region(
            &intent.mesh_summary,
            intent.driving_condition_region_id.as_deref(),
            intent
                .diagram_observation
                .as_ref()
                .and_then(|observation| observation.driving_condition_region_id.as_deref()),
            boundary_condition_region_id.as_deref(),
        )
        .map_err(|message| {
            author_study_error(
                &context,
                "RM.FEA.AUTHOR_STUDY.DRIVING_CONDITION_REGION_UNAVAILABLE",
                message,
                BTreeMap::from([("mesh_id".to_string(), intent.mesh_summary.mesh_id.clone())]),
            )
        })?;
        (Some(region_id), Some(source))
    };
    if let Some(boundary) = model.boundary_conditions.first_mut() {
        if let Some(region_id) = &boundary_condition_region_id {
            boundary.region_id = region_id.clone();
        }
    }
    let mut selected_structural_force_n = None;
    if let Some(load) = model.loads.first_mut() {
        if let Some(region_id) = &driving_condition_region_id {
            load.region_id = region_id.clone();
        }
        if let LoadKind::Force { fx, fy, fz } = &mut load.kind {
            let force_n = intent
                .structural_force_n
                .or_else(|| {
                    intent
                        .diagram_observation
                        .as_ref()
                        .and_then(|observation| observation.structural_force_n)
                })
                .unwrap_or([0.0, -1000.0, 0.0]);
            *fx = force_n[0];
            *fy = force_n[1];
            *fz = force_n[2];
            selected_structural_force_n = Some(force_n);
        }
    }
    let selected_driving_condition_kind = model
        .loads
        .first()
        .map(|load| load_kind_label(&load.kind).to_string());

    let study = AnalysisStudySpec {
        study_id: intent.study_id,
        geometry: intent.geometry,
        create_model_intent,
        model: Some(model),
        run_kind: intent.run_kind,
        backend: intent.backend,
        meshing_settings: Some(MeshingRequestSettings::default()),
        outputs: Vec::new(),
        solver_mesh_artifact_path: intent.solver_mesh_artifact_path.clone(),
        meshing_evidence_artifact_path: intent.meshing_evidence_artifact_path.clone(),
        linear_static_run_options: None,
        modal_run_options: None,
        acoustic_run_options: None,
        thermal_run_options: None,
        transient_run_options: None,
        cfd_run_options: None,
        cht_run_options: None,
        fsi_run_options: None,
        nonlinear_run_options: None,
        electromagnetic_run_options: None,
    };
    let issue_codes = validate_study_issue_codes(&study);
    if !issue_codes.is_empty() {
        return Err(author_study_error(
            &context,
            "RM.FEA.AUTHOR_STUDY.INVALID_AUTHORED_STUDY",
            "authored study failed validation",
            BTreeMap::from([("issue_codes".to_string(), issue_codes.join(","))]),
        ));
    }

    let evidence = AnalysisStudyAuthoringEvidence {
        schema_version: "fea_study_authoring_evidence/v1".to_string(),
        mesh_id: intent.mesh_summary.mesh_id.clone(),
        mesh_summary_schema_version: intent.mesh_summary.schema_version,
        selected_physical_region_id: physical_region_id,
        selected_boundary_condition_region_id: boundary_condition_region_id,
        selected_driving_condition_region_id: driving_condition_region_id,
        selected_driving_condition_kind,
        selected_structural_force_n,
        diagram_artifact_path: intent
            .diagram_observation
            .as_ref()
            .and_then(|observation| observation.artifact_path.clone()),
        diagram_source_mime_type: intent
            .diagram_observation
            .as_ref()
            .and_then(|observation| observation.source_mime_type.clone()),
        diagram_summary: intent
            .diagram_observation
            .as_ref()
            .and_then(|observation| observation.summary.clone()),
        diagram_confidence: intent
            .diagram_observation
            .as_ref()
            .and_then(|observation| observation.confidence),
        solver_mesh_artifact_path: intent.solver_mesh_artifact_path,
        meshing_evidence_artifact_path: intent.meshing_evidence_artifact_path,
        physical_region_source,
        boundary_condition_region_source,
        driving_condition_region_source,
    };
    let study_fingerprint = study_fingerprint(&study);
    let evidence_artifact_path = persist_study_evidence(
        &study_fingerprint,
        "author",
        serde_json::json!({
            "schema_version": "fea_study_authoring_artifact/v1",
            "study_id": study.study_id.clone(),
            "study_fingerprint": study_fingerprint,
            "evidence": evidence,
            "study": study,
        }),
    )
    .map_err(|err| {
        author_study_error(
            &context,
            "RM.FEA.AUTHOR_STUDY.ARTIFACT_STORE_FAILED",
            format!("failed to persist study authoring evidence artifact: {err}"),
            BTreeMap::from([("study_id".to_string(), study.study_id.clone())]),
        )
    })?;

    Ok(OperationEnvelope::new(
        ANALYSIS_AUTHOR_STUDY_OPERATION,
        ANALYSIS_AUTHOR_STUDY_OP_VERSION,
        &context,
        AnalysisStudyAuthoringData {
            study,
            evidence,
            evidence_artifact_path,
        },
    ))
}

fn load_kind_label(load: &LoadKind) -> &'static str {
    match load {
        LoadKind::Force { .. } => "force",
        LoadKind::Moment { .. } => "moment",
        LoadKind::Wrench { .. } => "wrench",
        LoadKind::Pressure { .. } => "pressure",
        LoadKind::BodyForce { .. } => "body_force",
        LoadKind::CurrentDensity { .. } => "current_density",
        LoadKind::CoilCurrent { .. } => "coil_current",
        LoadKind::HeatSource { .. } => "heat_source",
    }
}

fn author_study_error(
    context: &OperationContext,
    error_code: &'static str,
    message: impl Into<String>,
    context_values: BTreeMap<String, String>,
) -> OperationErrorEnvelope {
    operation_error(
        ANALYSIS_AUTHOR_STUDY_OPERATION,
        ANALYSIS_AUTHOR_STUDY_OP_VERSION,
        context,
        OperationErrorSpec {
            error_code,
            error_type: OperationErrorType::Validation,
            retryable: false,
            severity: OperationErrorSeverity::Error,
        },
        message,
        context_values,
    )
}

fn select_authoring_physical_region(
    summary: &super::AnalysisMeshSummary,
    requested: Option<&str>,
    diagram_observation: Option<&AnalysisStudyDiagramObservation>,
) -> Result<(String, String), String> {
    if let Some(requested) = requested {
        let Some(region) = summary
            .regions
            .physical_regions
            .iter()
            .find(|region| region.region_id == requested && region.element_count > 0)
        else {
            return Err(format!(
                "requested physical region `{requested}` is not available in mesh summary"
            ));
        };
        return Ok((region.region_id.clone(), "requested".to_string()));
    }

    if let Some(region_id) =
        diagram_observation.and_then(|observation| observation.physical_region_id.as_deref())
    {
        let Some(region) = summary
            .regions
            .physical_regions
            .iter()
            .find(|region| region.region_id == region_id && region.element_count > 0)
        else {
            return Err(format!(
                "diagram-selected physical region `{region_id}` is not available in mesh summary"
            ));
        };
        return Ok((region.region_id.clone(), "diagram".to_string()));
    }

    summary
        .regions
        .physical_regions
        .iter()
        .find(|region| region.element_count > 0)
        .map(|region| (region.region_id.clone(), "available".to_string()))
        .ok_or_else(|| "mesh summary has no physical regions".to_string())
}

fn select_authoring_boundary_region(
    summary: &super::AnalysisMeshSummary,
    requested: Option<&str>,
    diagram_region_id: Option<&str>,
    avoid_region_id: Option<&str>,
) -> Result<(String, String), String> {
    if let Some(requested) = requested {
        let Some(region) = summary.regions.boundary_regions.iter().find(|region| {
            region.region_id == requested && region.face_count > 0 && region.fully_recovered
        }) else {
            return Err(format!(
                "requested boundary region `{requested}` is not available and fully recovered in mesh authoring evidence"
            ));
        };
        return Ok((region.region_id.clone(), "requested".to_string()));
    }

    if let Some(region_id) = diagram_region_id {
        if avoid_region_id != Some(region_id) {
            let Some(region) = summary.regions.boundary_regions.iter().find(|region| {
                region.region_id == region_id && region.face_count > 0 && region.fully_recovered
            }) else {
                return Err(format!(
                    "diagram-selected boundary region `{region_id}` is not available and fully recovered in mesh authoring evidence"
                ));
            };
            return Ok((region.region_id.clone(), "diagram".to_string()));
        }
    }

    if let Some(region) = summary.regions.boundary_regions.iter().find(|region| {
        avoid_region_id != Some(region.region_id.as_str())
            && region.face_count > 0
            && region.fully_recovered
    }) {
        return Ok((region.region_id.clone(), "available".to_string()));
    }
    summary
        .regions
        .boundary_regions
        .iter()
        .find(|region| region.face_count > 0 && region.fully_recovered)
        .map(|region| (region.region_id.clone(), "available".to_string()))
        .ok_or_else(|| {
            "mesh authoring evidence has no fully recovered boundary regions".to_string()
        })
}
