use super::*;
use runmat_execution::value::{artifact_object_value_id, ValueRefKind};

fn reference(bytes: &[u8]) -> ValueRef {
    let digest = Digest::sha256(bytes);
    let context = Digest::sha256(b"catalog-object-test-authority");
    ValueRef {
        schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
        id: artifact_object_value_id(digest, context, "catalog-object-test"),
        logical_digest: digest,
        encoded_length: bytes.len() as u64,
        media_type: "application/vnd.runmat.analysis-chunk".into(),
        value_schema: "catalog-object-test/v1".into(),
        encryption_context: context,
        kind: ValueRefKind::ResultObject,
        authorization_scope: "catalog-object-test".into(),
        resident_fence: None,
    }
}

fn builder(root: &std::path::Path, definition: Digest, maximum: u64) -> AnalysisPublicationBuilder {
    let mut builder = AnalysisPublicationBuilder::new(definition, maximum).unwrap();
    builder.evidence_access =
        crate::analysis::structural_study::publication::evidence::StudyEvidenceAccess::new(
            root.to_owned(),
            Arc::new(runmat_filesystem::NativeFsProvider),
        );
    builder
}

#[test]
fn original_binary_object_survives_writer_drop_and_exact_retry_without_numeric_expansion() {
    let directory = tempfile::tempdir().unwrap();
    let definition = Digest::sha256(b"binary-catalog-publication");
    let bytes = (0..65_536)
        .map(|index| (index % 256) as u8)
        .collect::<Vec<_>>();
    let reference = reference(&bytes);
    // Original bytes and the possible existing immutable collision buffer coexist.
    let maximum = 2 * bytes.len() as u64;
    let original = builder(directory.path(), definition, maximum);
    let path = original
        .persist_catalog_object(definition, &reference, &bytes)
        .unwrap();
    assert_eq!(
        original
            .persist_catalog_object(definition, &reference, &bytes)
            .unwrap(),
        path
    );
    drop(original);
    let cold = runmat_filesystem::NativeFsProvider
        .read_bounded_sync(std::path::Path::new(&path), reference.encoded_length)
        .unwrap();
    assert_eq!(cold, bytes);
    assert_eq!(Digest::sha256(&cold), reference.logical_digest);
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        reference.encoded_length
    );
    let reopened = builder(directory.path(), definition, maximum);
    assert_eq!(
        reopened
            .persist_catalog_object(definition, &reference, &bytes)
            .unwrap(),
        path
    );
    std::fs::write(&path, b"corrupt existing object").unwrap();
    assert!(reopened
        .persist_catalog_object(definition, &reference, &bytes)
        .is_err());
    assert_eq!(std::fs::read(&path).unwrap(), b"corrupt existing object");
}

#[test]
fn binary_publication_rejects_foreign_scope_substitution_length_and_buffer_exhaustion() {
    let directory = tempfile::tempdir().unwrap();
    let definition = Digest::sha256(b"binary-catalog-rejection");
    let bytes = vec![7; 1024];
    let reference = reference(&bytes);
    let mut publication = builder(directory.path(), definition, 2 * bytes.len() as u64);
    assert!(publication
        .persist_catalog_object(Digest::sha256(b"foreign"), &reference, &bytes)
        .is_err());
    assert!(publication
        .persist_catalog_object(definition, &reference, &vec![8; bytes.len()])
        .is_err());
    let mut wrong_length = reference.clone();
    wrong_length.encoded_length += 1;
    assert!(publication
        .persist_catalog_object(definition, &wrong_length, &bytes)
        .is_err());
    publication.maximum_bytes -= 1;
    assert!(matches!(
        publication.persist_catalog_object(definition, &reference, &bytes),
        Err(AnalysisPublicationError::ResourceLimit { .. })
    ));
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
    publication.maximum_bytes += 1;
    assert!(publication
        .persist_catalog_object(definition, &reference, &bytes)
        .is_ok());
}
