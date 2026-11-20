//! Stored records name their format, and the format names one shape. A stored run record is
//! authenticated by re-serializing its typed metadata, so any field added, removed or renamed
//! in a persisted type makes older records unreadable. These tests pin the shape of the records
//! the fixture runs store against the record format versions; a shape change without a version
//! bump fails here instead of reaching users as a digest mismatch.
//!
//! The shape lists every struct the fixtures reach with all its field names, including fields
//! `skip_serializing_if` leaves out of the bytes, so an added optional field trips the guard.
use super::*;
use crate::analysis::storage::record_format::shape::{
    RecordShape, FREQUENCY_RECORD_VERSIONS, RUN_RECORD_VERSION,
};

/// (ARTIFACT_SCHEMA_VERSION, shape digest of the stored run record).
const PINNED_RUN_RECORD: (&str, &str) = (
    "analysis_run_artifact/2026-11-002",
    "sha256:1a3b48278be8364451993fff83e7442288d43a9bc6a1ec9ff99761d22c19c00f",
);

/// (frequency record versions, shape digest of the study step record they embed).
const PINNED_STEP_RECORD: ([&str; 2], &str) = (
    [
        "analysis_structural_frequency_artifact/2026-11-001",
        "analysis_acoustic_frequency_artifact/2026-11-001",
    ],
    "sha256:d75e9c564aff1353cceced6613678d479dfde2eb78f2efb182fcf0463cdda50e",
);

fn shape_failure(record: &str, constants: &str, pinned: &str, shape: &RecordShape) -> String {
    eprintln!("{record} shape:\n{}", shape.text());
    format!(
        "the {record} shape changed (digest {}, pinned {pinned}). Records stored by earlier \
         builds can no longer be authenticated. Bump {constants} to a new version and update \
         the pinned value in {} to the new version and digest.",
        shape.digest(),
        file!()
    )
}

/// A static solid run with a solver mesh (render topology and its regions), authored outputs
/// (observation evidence with its source field), and a modal and a transient run.
fn stored_run_shape() -> RecordShape {
    let (mesh_root, mesh_path) = write_ready_minimal_analysis_mesh_artifact("run-record-shape");
    let mut model = sample_solid_model();
    let step_id = model.sequence.steps[0].step_id.0.clone();
    let outputs = vec![
        super::static_assessments::displacement_peak(&step_id, &model.geometry_id),
        super::static_assessments::element_peak(&step_id, &model.geometry_id, "peak_stress"),
    ];
    runmat_analysis_core::attach_authored_outputs(&outputs, &mut model.sequence).unwrap();
    let options = AnalysisRunOptions {
        solver_mesh_artifact_path: Some(mesh_path.display().to_string()),
        ..AnalysisRunOptions::default()
    };
    let solid = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        options,
        OperationContext::new(None, None),
    )
    .unwrap()
    .data;
    // The stored record carries the scalar evidence of the run's observations.
    assert!(solid.render_topology.is_some());
    assert!(!solid.run.observations.is_empty());

    let mut modal_model = sample_model();
    modal_model.sequence.steps = vec![ProcedureStep::structural_modal("shape-modal")];
    let modal = analysis_run_modal_with_options_op(
        &modal_model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .unwrap()
    .data;
    let mut transient_model = sample_model();
    transient_model.sequence.steps = vec![ProcedureStep::structural_transient("shape-transient")];
    let transient = analysis_run_transient_with_options_op(
        &transient_model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .unwrap()
    .data;

    let mut shape = RecordShape::default();
    for run in [&solid, &modal, &transient] {
        // The stored record, and the run it was prepared from: content the record chunks out
        // of its metadata is decoded back into these types.
        shape.record_stored_run(run);
        shape.record(run);
    }
    let _ = fs::remove_dir_all(&mesh_root);
    shape
}

#[test]
fn stored_run_record_shape_is_pinned_to_its_format_version() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let shape = stored_run_shape();
    // The fixtures reach the nested types whose growth this guard exists for.
    for (owner, field) in [
        ("AnalysisRunResult", "render_topology"),
        ("AnalysisRunResult", "observation_scalars"),
        ("AnalysisRunResult", "modal_results"),
        ("AnalysisRunResult", "transient_results"),
        ("AnalysisRenderMesh", "regions"),
        ("ScalarObservationEvidence", "source_field_id"),
        ("PersistedRunArtifact", "metadata"),
    ] {
        assert!(
            shape.contains(owner, field),
            "fixture no longer reaches {owner}.{field}:\n{}",
            shape.text()
        );
    }
    let current = (RUN_RECORD_VERSION, shape.digest());
    assert!(
        current.0 == PINNED_RUN_RECORD.0 && current.1 == PINNED_RUN_RECORD.1,
        "{}",
        shape_failure(
            "stored run record",
            "ARTIFACT_SCHEMA_VERSION (analysis/storage.rs)",
            PINNED_RUN_RECORD.1,
            &shape
        )
    );
}

#[test]
fn study_step_record_shape_is_pinned_to_the_frequency_record_versions() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("step-record-shape");
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_study_artifact_root(&root);
    let mut spec = sample_linear_static_study_spec();
    let (mesh_root, mesh_path) = write_ready_minimal_analysis_mesh_artifact("step-record-shape");
    spec.solver_mesh_artifact_path = Some(mesh_path.display().to_string());
    let published = analysis_run_study_op(&spec, OperationContext::new(None, None))
        .expect("study run")
        .data;
    let mut shape = RecordShape::default();
    for step in &published.steps {
        assert!(step.render_topology.is_some());
        shape.record(step);
    }
    let _ = fs::remove_dir_all(&root);
    let _ = fs::remove_dir_all(&mesh_root);
    assert!(
        shape.contains("AnalysisStudyStepRunData", "render_topology"),
        "{}",
        shape.text()
    );
    let current = (FREQUENCY_RECORD_VERSIONS, shape.digest());
    assert!(
        current.0 == PINNED_STEP_RECORD.0 && current.1 == PINNED_STEP_RECORD.1,
        "{}",
        shape_failure(
            "study step record",
            "FREQUENCY_ARTIFACT_SCHEMA_VERSION and ACOUSTIC_FREQUENCY_ARTIFACT_SCHEMA_VERSION \
             (analysis/storage/frequency*.rs)",
            PINNED_STEP_RECORD.1,
            &shape
        )
    );
}
