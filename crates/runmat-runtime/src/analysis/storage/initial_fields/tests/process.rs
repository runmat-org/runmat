//! Original publication must remain readable after all producer memory is gone.
use super::*;
use std::path::Path;

const DIRECTORY_ENV: &str = "RUNMAT_ORIGINAL_FIELD_PROCESS_TEST_DIRECTORY";

#[derive(Serialize, Deserialize)]
struct SourceReceipt {
    reference: runmat_analysis_core::InitialFieldArtifactRef,
    value_digest: Digest,
    producer_pid: u32,
}

fn directory() -> std::path::PathBuf {
    std::env::var_os(DIRECTORY_ENV)
        .expect("process witness directory")
        .into()
}

fn spawn(entrypoint: &str, directory: &Path) {
    let (_, module) = module_path!().split_once("::").unwrap();
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            &format!("{module}::{entrypoint}"),
            "--nocapture",
        ])
        .env(DIRECTORY_ENV, directory)
        .output()
        .expect("original-field helper process must launch");
    assert!(
        output.status.success(),
        "helper {entrypoint}: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("1 passed"),
        "helper filter selected no witness"
    );
}

#[test]
fn original_source_survives_terminated_producer_and_fresh_consumer() {
    let directory = tempfile::tempdir().unwrap();
    spawn("original_source_producer_process", directory.path());
    // output() waits for terminal exit before the separate consumer can start.
    spawn("original_source_consumer_process", directory.path());
    let receipt: SourceReceipt =
        serde_json::from_slice(&std::fs::read(directory.path().join("source.json")).unwrap())
            .unwrap();
    let consumer_pid: u32 =
        serde_json::from_slice(&std::fs::read(directory.path().join("consumer.json")).unwrap())
            .unwrap();
    assert_ne!(receipt.producer_pid, consumer_pid);
    assert_ne!(receipt.producer_pid, std::process::id());
    assert_ne!(consumer_pid, std::process::id());
}

#[test]
#[ignore = "entrypoint launched by the process-termination witness"]
fn original_source_producer_process() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let root = directory();
    let (run, artifact) = source();
    let store = FilesystemAnalysisArtifactStore::new(root.join("store"));
    store
        .commit_publication(prepare(&run, &artifact.manifest().unwrap()))
        .unwrap();
    let field = &artifact.fields()[0];
    let reference = runmat_analysis_core::InitialFieldArtifactRef {
        artifact_id: run.run_id.clone(),
        field_id: field.definition().field_id().into(),
        content_digest: artifact.identity().digest().to_string(),
        topology_digest: artifact.topology_digest.to_string(),
    };
    let receipt = SourceReceipt {
        reference,
        value_digest: Digest::sha256(serde_json::to_vec(field.as_host_f64().unwrap()).unwrap()),
        producer_pid: std::process::id(),
    };
    // Only identities cross this boundary; original values stay in the shared store.
    std::fs::write(
        root.join("source.json"),
        serde_json::to_vec(&receipt).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "entrypoint launched by the process-termination witness"]
fn original_source_consumer_process() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let root = directory();
    let receipt: SourceReceipt =
        serde_json::from_slice(&std::fs::read(root.join("source.json")).unwrap()).unwrap();
    configure_artifact_store(AnalysisArtifactStoreConfig::Filesystem {
        root: root.join("store"),
    })
    .unwrap();
    let fields = resolve_initial_field_references(
        [&receipt.reference, &receipt.reference].into_iter(),
        LIMIT,
    )
    .unwrap();
    assert_eq!(fields.len(), 1, "shared references resolve once");
    assert_eq!(fields[0].reference(), &receipt.reference);
    assert_eq!(
        Digest::sha256(
            serde_json::to_vec(fields[0].artifact().fields()[0].as_host_f64().unwrap()).unwrap()
        ),
        receipt.value_digest
    );
    assert!(resolve_initial_field_references(std::iter::once(&receipt.reference), 1).is_err());
    let mut foreign = receipt.reference;
    foreign.content_digest = Digest::sha256("foreign source").to_string();
    assert!(resolve_initial_field_references(std::iter::once(&foreign), LIMIT).is_err());
    std::fs::write(
        root.join("consumer.json"),
        serde_json::to_vec(&std::process::id()).unwrap(),
    )
    .unwrap();
}
