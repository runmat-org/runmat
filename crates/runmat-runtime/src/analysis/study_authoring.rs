use std::collections::BTreeMap;

use runmat_analysis_core::{
    AnalysisStep, AnalysisStepKind, BoundaryCondition, BoundaryConditionKind, EvidenceConfidence,
    LoadCase, LoadKind, MaterialAssignment,
};
use runmat_meshing_core::VolumeMeshingOptions;

use crate::operations::{
    operation_error, OperationContext, OperationEnvelope, OperationErrorEnvelope,
    OperationErrorSeverity, OperationErrorSpec, OperationErrorType,
};

use super::{
    analysis_create_model_op, persist_study_evidence, profile_supports_run_kind, study_fingerprint,
    validate_study_issue_codes, AnalysisCreateModelIntentSpec, AnalysisCreateModelProfile,
    AnalysisRunKind, AnalysisStudyAuthoringData, AnalysisStudyAuthoringEvidence,
    AnalysisStudyAuthoringIntent, AnalysisStudySpec, ANALYSIS_AUTHOR_STUDY_OPERATION,
    ANALYSIS_AUTHOR_STUDY_OP_VERSION,
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
    if !intent.mesh_authoring_summary.solve_ready {
        return Err(author_study_error(
            &context,
            "RM.FEA.AUTHOR_STUDY.MESH_NOT_SOLVE_READY",
            "mesh authoring summary is not solve-ready",
            BTreeMap::from([
                (
                    "mesh_id".to_string(),
                    intent.mesh_authoring_summary.mesh_id.clone(),
                ),
                (
                    "validation_error_code".to_string(),
                    intent
                        .mesh_authoring_summary
                        .validation_error_code
                        .clone()
                        .unwrap_or_default(),
                ),
            ]),
        ));
    }
    if intent.profile != AnalysisCreateModelProfile::LinearStaticStructural
        || intent.run_kind != AnalysisRunKind::LinearStatic
    {
        return Err(author_study_error(
            &context,
            "RM.FEA.AUTHOR_STUDY.UNSUPPORTED_PROFILE",
            "compact evidence study authoring currently supports linear static structural studies",
            BTreeMap::from([
                ("profile".to_string(), format!("{:?}", intent.profile)),
                ("run_kind".to_string(), format!("{:?}", intent.run_kind)),
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

    let (material_region_id, material_region_source) = select_authoring_material_region(
        &intent.mesh_authoring_summary,
        intent.material_region_id.as_deref(),
    )
    .map_err(|message| {
        author_study_error(
            &context,
            "RM.FEA.AUTHOR_STUDY.MATERIAL_REGION_UNAVAILABLE",
            message,
            BTreeMap::from([(
                "mesh_id".to_string(),
                intent.mesh_authoring_summary.mesh_id.clone(),
            )]),
        )
    })?;
    let (fixed_boundary_region_id, fixed_boundary_region_source) =
        select_authoring_boundary_region(
            &intent.mesh_authoring_summary,
            intent.fixed_boundary_region_id.as_deref(),
            None,
        )
        .map_err(|message| {
            author_study_error(
                &context,
                "RM.FEA.AUTHOR_STUDY.FIXED_BOUNDARY_REGION_UNAVAILABLE",
                message,
                BTreeMap::from([(
                    "mesh_id".to_string(),
                    intent.mesh_authoring_summary.mesh_id.clone(),
                )]),
            )
        })?;
    let (load_boundary_region_id, load_boundary_region_source) = select_authoring_boundary_region(
        &intent.mesh_authoring_summary,
        intent.load_boundary_region_id.as_deref(),
        Some(fixed_boundary_region_id.as_str()),
    )
    .map_err(|message| {
        author_study_error(
            &context,
            "RM.FEA.AUTHOR_STUDY.LOAD_BOUNDARY_REGION_UNAVAILABLE",
            message,
            BTreeMap::from([(
                "mesh_id".to_string(),
                intent.mesh_authoring_summary.mesh_id.clone(),
            )]),
        )
    })?;

    let force_n = intent.force_n.unwrap_or([0.0, -1000.0, 0.0]);
    let model_id = intent
        .model_id
        .clone()
        .filter(|model_id| !model_id.trim().is_empty())
        .unwrap_or_else(|| format!("{}_model", intent.study_id.trim()));
    let create_model_intent = AnalysisCreateModelIntentSpec {
        model_id: model_id.clone(),
        profile: intent.profile,
        prep_context: None,
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
        region_id: material_region_id.clone(),
        expected_material_id: material_id.clone(),
        assigned_material_id: material_id,
        confidence: EvidenceConfidence::Inferred,
    }];
    model.boundary_conditions = vec![BoundaryCondition {
        bc_id: "bc_authored_fixed".to_string(),
        region_id: fixed_boundary_region_id.clone(),
        kind: BoundaryConditionKind::Fixed,
    }];
    model.loads = vec![LoadCase {
        load_id: "load_authored_force".to_string(),
        region_id: load_boundary_region_id.clone(),
        kind: LoadKind::Force {
            fx: force_n[0],
            fy: force_n[1],
            fz: force_n[2],
        },
    }];
    model.steps = vec![AnalysisStep {
        step_id: "step_authored_static".to_string(),
        kind: AnalysisStepKind::Static,
    }];

    let study = AnalysisStudySpec {
        study_id: intent.study_id,
        geometry: intent.geometry,
        create_model_intent,
        model: Some(model),
        run_kind: intent.run_kind,
        backend: intent.backend,
        mesh_options: Some(VolumeMeshingOptions::default()),
        analysis_mesh_artifact_path: intent.analysis_mesh_artifact_path.clone(),
        analysis_mesh_evidence_artifact_path: intent.analysis_mesh_evidence_artifact_path.clone(),
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
        mesh_id: intent.mesh_authoring_summary.mesh_id.clone(),
        mesh_authoring_summary_schema_version: intent.mesh_authoring_summary.schema_version.clone(),
        tetrahedron_generation_family: intent.mesh_authoring_summary.tetrahedron_generation_family,
        tetrahedron_generation_attempted_family_count: intent
            .mesh_authoring_summary
            .tetrahedron_generation_attempted_family_count,
        tetrahedron_generation_rejected_family_count: intent
            .mesh_authoring_summary
            .tetrahedron_generation_rejected_family_count,
        tetrahedron_generation_selected_family_index: intent
            .mesh_authoring_summary
            .tetrahedron_generation_selected_family_index,
        tetrahedron_generation_interior_support_candidate_count: intent
            .mesh_authoring_summary
            .tetrahedron_generation_interior_support_candidate_count,
        tetrahedron_generation_interior_support_accepted_count: intent
            .mesh_authoring_summary
            .tetrahedron_generation_interior_support_accepted_count,
        selected_material_region_id: material_region_id,
        selected_fixed_boundary_region_id: fixed_boundary_region_id,
        selected_load_boundary_region_id: load_boundary_region_id,
        selected_force_n: force_n,
        analysis_mesh_artifact_path: intent.analysis_mesh_artifact_path,
        analysis_mesh_evidence_artifact_path: intent.analysis_mesh_evidence_artifact_path,
        material_region_source,
        fixed_boundary_region_source,
        load_boundary_region_source,
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

fn select_authoring_material_region(
    summary: &runmat_meshing_evidence::MeshAuthoringSummary,
    requested: Option<&str>,
) -> Result<(String, String), String> {
    if let Some(requested) = requested {
        let Some(region) = summary
            .regions
            .material_regions
            .iter()
            .find(|region| region.region_id == requested && region.element_count > 0)
        else {
            return Err(format!(
                "requested material region `{requested}` is not available in mesh authoring evidence"
            ));
        };
        return Ok((region.region_id.clone(), "requested".to_string()));
    }

    for required in &summary.regions.required_material_region_ids {
        if summary
            .regions
            .material_regions
            .iter()
            .any(|region| region.region_id == *required && region.element_count > 0)
        {
            return Ok((required.clone(), "required".to_string()));
        }
    }
    summary
        .regions
        .material_regions
        .iter()
        .find(|region| region.element_count > 0)
        .map(|region| (region.region_id.clone(), "available".to_string()))
        .ok_or_else(|| "mesh authoring evidence has no material regions".to_string())
}

fn select_authoring_boundary_region(
    summary: &runmat_meshing_evidence::MeshAuthoringSummary,
    requested: Option<&str>,
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

    for required in &summary.regions.required_boundary_region_ids {
        if avoid_region_id == Some(required.as_str()) {
            continue;
        }
        if summary.regions.boundary_regions.iter().any(|region| {
            region.region_id == *required && region.face_count > 0 && region.fully_recovered
        }) {
            return Ok((required.clone(), "required".to_string()));
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
