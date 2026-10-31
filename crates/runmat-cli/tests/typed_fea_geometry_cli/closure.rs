//! Cold import of the exact binary objects referenced by a published catalog.
use runmat_execution::value::ValueRef;
use runmat_execution_artifact::{
    cache::{CacheExport, FilesystemObjectStore},
    LogicalObject, ObjectNamespace,
};
use serde_json::Value;
use tempfile::TempDir;

pub(super) fn reopen_catalog_objects(project: &TempDir, manifest: &Value) -> FilesystemObjectStore {
    let evidence = &manifest["numerical_evidence"];
    let paths = evidence["object_artifact_paths"].as_array().unwrap();
    let references: Vec<ValueRef> =
        serde_json::from_value(evidence["catalog"]["objects"].clone()).unwrap();
    assert_eq!(paths.len(), references.len());
    assert!(paths.len() > 3);
    let mut cold =
        FilesystemObjectStore::open(project.path().join("cold-closure"), 256 * 1024 * 1024)
            .unwrap();
    let mut digests = std::collections::BTreeSet::new();
    // Publication persists objects in catalog order. Their metadata lives in
    // the sealed manifest; each file contains the original object bytes.
    for (path, reference) in paths.iter().zip(references) {
        let bytes = std::fs::read(project.path().join(path.as_str().unwrap())).unwrap();
        assert_eq!(bytes.len() as u64, reference.encoded_length);
        assert_eq!(
            runmat_execution::Digest::sha256(&bytes),
            reference.logical_digest
        );
        assert!(digests.insert(reference.logical_digest));
        let logical = LogicalObject::new(
            ObjectNamespace::ResultValue,
            format!("cold/{}", reference.logical_digest),
            reference.media_type,
            bytes,
        )
        .unwrap();
        CacheExport::write_verified(&mut cold, &logical).unwrap();
    }
    cold
}
