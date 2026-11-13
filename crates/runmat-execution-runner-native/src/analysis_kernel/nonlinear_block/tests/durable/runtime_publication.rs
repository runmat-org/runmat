//! Publish the actual original cold source, then read history after publisher exit.
use super::*;
use runmat_analysis_execution::PreparedNonlinearSolidStudyRoot;
use runmat_runtime::analysis::{
    self, storage::*, AnalysisStudyCompletionIdentity, FeaRuntimeConfig,
};
use runmat_runtime::operations::{OperationContext, OperationErrorType};

pub(crate) mod history;

const ROOT_ENV: &str = "RUNMAT_J2_RUNTIME_PUBLICATION_ROOT";

pub(super) fn verify(
    source_root: &std::path::Path,
    prepared: &PreparedNonlinearSolidStudyRoot,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    policy: AnalysisOutputPolicy,
    ceiling: u64,
    expected_factors: &[f64],
    reference: history::Reference,
) {
    let limits = policy.inventory_limits();
    std::fs::write(
        source_root.join("handoff.json"),
        serde_json::to_vec(&serde_json::json!({
            "root": prepared.root, "mesh": mesh, "ceiling": ceiling,
            "max_objects": limits.max_objects, "max_object_bytes": limits.max_object_bytes,
            "max_total_bytes": limits.max_total_bytes,
        }))
        .unwrap(),
    )
    .unwrap();
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            &format!("{}::publisher", module_path!().split_once("::").unwrap().1),
            "--exact",
            "--ignored",
            "--nocapture",
        ])
        .env(ROOT_ENV, source_root)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "publisher failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let completion: analysis::AnalysisStudyRunData =
        serde_json::from_slice(&std::fs::read(source_root.join("published.json")).unwrap())
            .unwrap();
    assert_eq!(completion.states.len(), 3);
    let store = FilesystemAnalysisArtifactStore::new(source_root.join("runs"));
    let run_id = &completion.steps[0].run_id;
    let original = store.load_run(run_id).unwrap().unwrap();
    let history = original.nonlinear_results.unwrap();
    assert_eq!(history.load_factors, expected_factors);
    let accepted_count = expected_factors.len();
    assert_eq!(history.load_factors.last(), Some(&1.0));
    assert_eq!(history.displacement_snapshots.len(), accepted_count);
    assert_eq!(history.plastic_strain_snapshots.len(), accepted_count);
    history::verify(reference, mesh, &history, &original.run, &store, run_id);
    assert!(store
        .load_publication_manifest_path(
            std::path::Path::new(completion.result_manifest_artifact_path.as_ref().unwrap()),
            ceiling,
        )
        .is_ok());
}

#[test]
#[ignore = "invoked by the original-root witness in an isolated publisher process"]
fn publisher() {
    let directory = std::path::PathBuf::from(std::env::var_os(ROOT_ENV).unwrap());
    let wire: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("handoff.json")).unwrap()).unwrap();
    let root: runmat_execution::value::ValueRef =
        serde_json::from_value(wire["root"].clone()).unwrap();
    let mesh: runmat_meshing_core::SolverMeshArtifact =
        serde_json::from_value(wire["mesh"].clone()).unwrap();
    let ceiling = wire["ceiling"].as_u64().unwrap();
    let limits = runmat_execution_artifact::object::ObjectInventoryLimits {
        max_objects: usize::try_from(wire["max_objects"].as_u64().unwrap()).unwrap(),
        max_object_bytes: wire["max_object_bytes"].as_u64().unwrap(),
        max_total_bytes: wire["max_total_bytes"].as_u64().unwrap(),
    };
    let access = ArtifactAccess {
        authorization_scope: root.authorization_scope.clone(),
        encryption_context: root.encryption_context,
    };
    let source = FilesystemObjectStore::open(&directory, limits.max_object_bytes).unwrap();
    let (original, source) =
        crate::analysis_nonlinear::source::nonlinear_solid_study_completion_from_source(
            source,
            root.clone(),
            root.logical_digest,
            access,
            limits,
            ceiling,
        )
        .unwrap();
    let config = FeaRuntimeConfig {
        artifact_root: Some(directory.join("artifacts")),
        study_artifact_root: Some(directory.join("studies")),
        ..Default::default()
    };
    analysis::configure_fea_runtime(FeaRuntimeConfig {
        publication_staging_bytes: 1,
        ..config.clone()
    })
    .unwrap();
    configure_artifact_store(AnalysisArtifactStoreConfig::Filesystem {
        root: directory.join("runs"),
    })
    .unwrap();
    let store = FilesystemAnalysisArtifactStore::new(directory.join("runs"));
    let evidence = runmat_meshing_core::fixtures::canonical_tetrahedron_meshing_evidence(&mesh);
    let definition = original.compiled.definition_digest().unwrap();
    let identity = AnalysisStudyCompletionIdentity::new("original-j2-cold-runtime").unwrap();
    let prepare = || {
        analysis::prepare_nonlinear_solid_study_publication(
            &original,
            &source,
            &mesh,
            &evidence,
            definition,
            OperationContext::new(None, None),
            identity.clone(),
        )
    };
    let rejected = prepare()
        .err()
        .expect("insufficient host preparation must reject");
    assert_eq!(rejected.error_type, OperationErrorType::Capacity);
    assert!(store.list_run_metadata().unwrap().is_empty());
    analysis::configure_fea_runtime(config).unwrap();
    let prepared = prepare().unwrap();
    assert!(store.list_run_metadata().unwrap().is_empty());
    let completed = prepared.publish_without_function_execution().unwrap().data;
    std::fs::write(
        directory.join("published.json"),
        serde_json::to_vec(&completed).unwrap(),
    )
    .unwrap();
}
