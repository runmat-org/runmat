//! The original admitted typed-study path, then cold reads after the producer process exits.
use super::*;
use runmat_analysis_fea::results::assessment::ScalarObservationEvidence;
use runmat_runtime::analysis::{
    storage::{
        AnalysisArtifactStore, AnalysisArtifactStoreConfig, AnalysisFieldPageLookup,
        AssessmentStoreError, FilesystemAnalysisArtifactStore,
    },
    AnalysisFieldRequestOptions, AnalysisFieldSelection, AnalysisStudyCompletionIdentity,
    FeaRuntimeConfig, StructuralStudyFrequencyNumericalResult, StructuralStudyNumericalResult,
    StructuralStudyResult,
};
use runmat_runtime::operations::OperationContext;

const ROOT_ENV: &str = "RUNMAT_TEST_FREQUENCY_PUBLICATION_ROOT";
#[path = "publication/cancellation.rs"]
mod cancellation;
#[path = "publication/crash.rs"]
mod crash;
const PRODUCER: &str =
    "analysis_kernel::frequency_tests::cold_history::publication::frequency_publication_producer";

#[test]
fn original_frequency_publication_survives_producer_exit_and_cold_runtime_reads() {
    let directory = tempfile::tempdir().unwrap();
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([PRODUCER, "--exact", "--ignored", "--nocapture"])
        .env(ROOT_ENV, directory.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "original producer failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        verify_cold(&directory.path().join(format!("{order:?}")));
    }
}

#[test]
#[ignore = "invoked in an isolated process by the cold-runtime publication witness"]
fn frequency_publication_producer() {
    let root = std::path::PathBuf::from(std::env::var_os(ROOT_ENV).expect("isolated witness root"));
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        if !crash::selected(order) {
            continue;
        }
        super::check(order, Some(&root.join(format!("{order:?}"))));
    }
}

pub(super) fn publish(
    root: &std::path::Path,
    order: ElementOrder,
    mesh: runmat_meshing_core::SolverMeshArtifact,
    numerical: StructuralStudyFrequencyNumericalResult,
    scalars: Vec<ScalarObservationEvidence>,
    series: ScalarObservationEvidence,
    original_pages: Vec<runmat_runtime::analysis::AnalysisFieldPageValues>,
) {
    std::fs::create_dir_all(root).unwrap();
    let runs = root.join("runs");
    runmat_runtime::analysis::configure_fea_runtime(FeaRuntimeConfig {
        artifact_root: Some(root.join("artifacts")),
        study_artifact_root: Some(root.join("studies")),
        ..Default::default()
    })
    .unwrap();
    runmat_runtime::analysis::storage::configure_artifact_store(
        AnalysisArtifactStoreConfig::Filesystem { root: runs.clone() },
    )
    .unwrap();
    let compiled = runmat_analysis_fea::study::CompiledAnalysisStudy {
        module_path: "original-native-frequency.fea".into(),
        study_id: "original-frequency".into(),
        study_inputs: Vec::new(),
        outputs: Vec::new(),
        assessments: Vec::new(),
        model: numerical.workload.model().clone(),
        properties: runmat_analysis_fea::properties::compile_property_map(
            numerical.workload.model(),
            Some(&mesh),
        )
        .unwrap(),
        solver_mesh_digest: mesh.canonical_digest,
    };
    let workloads = runmat_analysis_fea::execution::StructuralStudyWorkloads::StructuralFrequency {
        workload: Box::new(numerical.workload.clone()),
    };
    let evidence = runmat_meshing_core::fixtures::canonical_tetrahedron_meshing_evidence(&mesh);
    let result = StructuralStudyResult::new(
        compiled,
        mesh,
        evidence,
        workloads,
        StructuralStudyNumericalResult::StructuralFrequency(Box::new(numerical)),
    )
    .unwrap();
    let StructuralStudyNumericalResult::StructuralFrequency(frequency) = result.numerical() else {
        unreachable!()
    };
    let scalar_object = frequency
        .artifact_roots()
        .samples()
        .last()
        .unwrap()
        .observations()
        .unwrap()
        .reference()
        .logical_digest
        .bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let field_id = frequency.publication.samples()[0].fields.fields[0]
        .definition
        .field_id()
        .to_owned();
    runmat_runtime::analysis::configure_fea_runtime(FeaRuntimeConfig {
        artifact_root: Some(root.join("artifacts")),
        study_artifact_root: Some(root.join("studies")),
        publication_staging_bytes: 1,
    })
    .unwrap();
    let identity =
        AnalysisStudyCompletionIdentity::new(format!("original-frequency-{order:?}")).unwrap();
    let rejected = runmat_runtime::analysis::prepare_structural_study_publication(
        &result,
        OperationContext::new(None, None),
        identity.clone(),
    )
    .err()
    .expect("insufficient host publication allowance must reject");
    assert_eq!(
        rejected.error_type,
        runmat_runtime::operations::OperationErrorType::Capacity
    );
    let store = FilesystemAnalysisArtifactStore::new(runs.clone());
    assert!(store.list_run_metadata().unwrap().is_empty());
    runmat_runtime::analysis::configure_fea_runtime(FeaRuntimeConfig {
        artifact_root: Some(root.join("artifacts")),
        study_artifact_root: Some(root.join("studies")),
        ..Default::default()
    })
    .unwrap();
    let prepared = runmat_runtime::analysis::prepare_structural_study_publication(
        &result,
        OperationContext::new(None, None),
        identity.clone(),
    )
    .unwrap();
    assert!(
        store.list_run_metadata().unwrap().is_empty(),
        "prepared frequency children remain private"
    );
    crash::wait_for_termination();
    let prepared = cancellation::reject_and_retry(prepared, &result, identity, &store);
    let redirected = root.join("redirected");
    runmat_runtime::analysis::storage::configure_artifact_store(
        AnalysisArtifactStoreConfig::Filesystem {
            root: redirected.clone(),
        },
    )
    .unwrap();
    let completed = prepared.publish_without_function_execution().unwrap().data;
    assert!(FilesystemAnalysisArtifactStore::new(redirected)
        .list_run_metadata()
        .unwrap()
        .is_empty());
    let [step] = completed.steps.as_slice() else {
        panic!("singleton frequency study")
    };
    let expected = scalars
        .into_iter()
        .filter(|scalar| scalar.observation_id.0 == "displacement_peak")
        .chain([series])
        .collect::<Vec<_>>();
    assert_eq!(expected.len(), 2);
    let store = FilesystemAnalysisArtifactStore::new(runs);
    let metadata = store
        .load_assessment_metadata(&step.run_id, 64 << 20)
        .unwrap()
        .unwrap();
    assert_eq!(metadata.observation_scalars, expected);
    let mut pages = Vec::new();
    assert_eq!(original_pages.len(), 2);
    for (frequency, original_values) in [20.0, 40.0].into_iter().zip(original_pages) {
        let selection = AnalysisFieldSelection::new(
            field_id.clone(),
            Some(
                runmat_analysis_fea::results::field::RecoveredFieldSample::FrequencyHertz(
                    frequency,
                ),
            ),
        );
        let AnalysisFieldPageLookup::Page(page) = store
            .load_field_page(
                &step.run_id,
                &selection,
                AnalysisFieldRequestOptions::default(),
            )
            .unwrap()
        else {
            panic!("original frequency field page is missing")
        };
        assert_eq!(page.values, original_values);
        pages.push(serde_json::json!({"selection": selection, "page": page}));
    }
    std::fs::write(
        root.join("expected.json"),
        serde_json::to_vec(&serde_json::json!({
            "run_id": step.run_id, "identity": step.identity, "scalars": expected,
                "manifest_path": completed.result_manifest_artifact_path.unwrap(), "pages": pages,
                "scalar_object": scalar_object,
        }))
        .unwrap(),
    )
    .unwrap();
}

fn verify_cold(root: &std::path::Path) {
    let expected: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("expected.json")).unwrap()).unwrap();
    let run_id = expected["run_id"].as_str().unwrap();
    let store = FilesystemAnalysisArtifactStore::new(root.join("runs"));
    assert!(matches!(
        store.load_assessment_metadata(run_id, 0),
        Err(AssessmentStoreError::ResourceLimit)
    ));
    let metadata = store
        .load_assessment_metadata(run_id, 64 << 20)
        .unwrap()
        .unwrap();
    assert_eq!(
        serde_json::to_value(metadata.identity).unwrap(),
        expected["identity"]
    );
    assert_eq!(
        serde_json::to_value(metadata.observation_scalars).unwrap(),
        expected["scalars"]
    );
    let path = std::path::Path::new(expected["manifest_path"].as_str().unwrap());
    assert!(matches!(
        store.load_publication_manifest_path(path, 0),
        Err(AssessmentStoreError::ResourceLimit)
    ));
    assert!(store.load_publication_manifest_path(path, 64 << 20).is_ok());
    for expected_page in expected["pages"].as_array().unwrap() {
        let selection: AnalysisFieldSelection =
            serde_json::from_value(expected_page["selection"].clone()).unwrap();
        let AnalysisFieldPageLookup::Page(page) = store
            .load_field_page(run_id, &selection, AnalysisFieldRequestOptions::default())
            .unwrap()
        else {
            panic!("cold original field page is missing")
        };
        assert_eq!(serde_json::to_value(page).unwrap(), expected_page["page"]);
    }
    // A cold scalar decode must authenticate its original catalog root, even after earlier reads.
    let object = root
        .join("runs/objects")
        .join(expected["scalar_object"].as_str().unwrap());
    let original = std::fs::read(&object).unwrap();
    let mut substituted = original.clone();
    substituted[0] ^= 1;
    std::fs::write(&object, substituted).unwrap();
    assert!(matches!(
        store.load_assessment_metadata(run_id, 64 << 20),
        Err(AssessmentStoreError::Invalid(_))
    ));
    std::fs::write(&object, original).unwrap();
    assert!(store
        .load_assessment_metadata(run_id, 64 << 20)
        .unwrap()
        .is_some());
    // Cached children cannot outlive their one committed closed inventory.
    let marker = path.parent().unwrap().join("commit.json");
    let original_marker = std::fs::read(&marker).unwrap();
    std::fs::remove_file(&marker).unwrap();
    assert!(store
        .load_assessment_metadata(run_id, 64 << 20)
        .unwrap()
        .is_none());
    let selection: AnalysisFieldSelection =
        serde_json::from_value(expected["pages"][0]["selection"].clone()).unwrap();
    assert!(matches!(
        store
            .load_field_page(run_id, &selection, AnalysisFieldRequestOptions::default())
            .unwrap(),
        AnalysisFieldPageLookup::RunNotFound
    ));
    std::fs::write(marker, original_marker).unwrap();
    assert!(store
        .load_assessment_metadata(run_id, 64 << 20)
        .unwrap()
        .is_some());
}
