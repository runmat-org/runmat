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

/// One law: the semantic state persistence requirement is the larger of the encoder's
/// workspace (output included) and the immutable buffer plus its comparison. Persisting the
/// state's bytes at exactly that allowance succeeds, also on an existing-content retry, and one
/// byte less is refused before anything is written.
#[test]
fn state_persistence_allowance_is_the_immutable_batch_law_at_exactly_its_bound() {
    use crate::analysis::structural_study::publication::evidence::immutable_batch_workspace;
    let bytes = vec![3_u8; 4096];
    let len = bytes.len() as u64;
    let lean = runmat_canonical_codec::CanonicalEncodingProfile {
        encoded_bytes_upper_bound: len,
        resident_workspace_bytes_estimate: len + 64,
    };
    let bound = resources::state_persistence_workspace(&lean).unwrap();
    assert_eq!(bound, immutable_batch_workspace(len).unwrap());
    assert_eq!(bound, 2 * len);
    let heavy = runmat_canonical_codec::CanonicalEncodingProfile {
        encoded_bytes_upper_bound: len,
        resident_workspace_bytes_estimate: 3 * len,
    };
    assert_eq!(
        resources::state_persistence_workspace(&heavy).unwrap(),
        3 * len
    );

    let directory = tempfile::tempdir().unwrap();
    let definition = Digest::sha256(b"state-persistence-bound");
    let publication = builder(directory.path(), definition, bound);
    assert!(matches!(
        publication
            .evidence_access
            .persist_bytes("states/below.state", &bytes, bound - 1),
        Err(AnalysisPublicationError::ResourceLimit { .. })
    ));
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
    let path = publication
        .evidence_access
        .persist_bytes("states/at.state", &bytes, bound)
        .unwrap();
    assert_eq!(
        publication
            .evidence_access
            .persist_bytes("states/at.state", &bytes, bound)
            .unwrap(),
        path
    );
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
}

/// Admission funds catalog persistence from the same law: a source whose own profiled demand
/// is only its transport is still leased the buffer plus its comparison.
#[test]
fn profiled_source_admission_funds_the_immutable_batch_law() {
    use crate::analysis::structural_study::publication::evidence::immutable_batch_workspace;
    let directory = tempfile::tempdir().unwrap();
    let definition = Digest::sha256(b"profiled-source-persistence");
    let bytes = vec![5_u8; 2048];
    let reference = reference(&bytes);
    let len = bytes.len() as u64;
    let mut publication = builder(directory.path(), definition, 2 * len);
    publication
        .admit_profiled_source_objects(std::slice::from_ref(&reference), len)
        .unwrap();
    assert_eq!(
        publication.source_workspace_bytes,
        immutable_batch_workspace(len).unwrap()
    );
    assert!(resources::source_workspace(std::slice::from_ref(&reference)).unwrap() >= 2 * len);
    let _phase = publication.begin_source_phase().unwrap();
    publication
        .persist_catalog_objects(definition, std::slice::from_ref(&reference), |_| {
            Ok(bytes.clone())
        })
        .unwrap();
}
