//! Original bytes and source counters discriminate allocation and authentication failures.
use super::*;
use runmat_analysis_fea::execution::AnalysisChunkDescriptor;
use runmat_execution::value::{artifact_object_value_id, ValueRef, ValueRefKind};
use std::cell::Cell;

fn catalog(bytes: &[u8]) -> PublishedObjectCatalog {
    let logical_digest = Digest::sha256(bytes);
    let context = Digest::sha256(b"assessment-source");
    let reference = ValueRef {
        schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
        id: artifact_object_value_id(logical_digest, context, "assessment-test"),
        logical_digest,
        encoded_length: bytes.len() as u64,
        media_type: "application/vnd.runmat.test".into(),
        value_schema: "runmat.test".into(),
        encryption_context: context,
        kind: ValueRefKind::ResultObject,
        authorization_scope: "assessment-test".into(),
        resident_fence: None,
    };
    PublishedObjectCatalog::new(vec![reference.clone()], vec![reference]).unwrap()
}
fn chunk(bytes: &[u8]) -> AnalysisChunkDescriptor {
    AnalysisChunkDescriptor {
        ordinal: 0,
        digest: Digest::sha256(bytes),
        encoded_length: bytes.len() as u64,
    }
}
#[test]
fn forged_payload_capacity_and_coexisting_chunk_reject_before_source_read() {
    let bytes = b"ORIGINAL";
    let catalog = catalog(bytes);
    let reads = Cell::new(0);
    let mut reader = resources::Reader::new(&catalog, 1 << 30, 1024, |_, _| {
        reads.set(reads.get() + 1);
        Ok(Some(bytes.to_vec()))
    });
    for (encoded, retained) in [(1 << 30, 0), (8, 1016)] {
        assert!(matches!(
            reader.payload(&[chunk(bytes)], encoded, Digest::sha256(bytes), retained),
            Err(AssessmentStoreError::ResourceLimit)
        ));
        assert_eq!(reads.get(), 0);
    }
    assert_eq!(
        reader
            .payload(&[chunk(bytes)], 8, Digest::sha256(bytes), 0)
            .unwrap(),
        bytes
    );
    assert_eq!(reads.get(), 1);
}
#[test]
fn canonical_manifest_shape_is_admitted_before_typed_decode() {
    let bytes = runmat_canonical_codec::encode_contract(
        b"assessment-test\0",
        "dense",
        &vec![1u64; 8],
        runmat_canonical_codec::CanonicalLimits::new(1024, 100, 100, 16),
    )
    .unwrap();
    let catalog = catalog(&bytes);
    let reads = Cell::new(0);
    let mut reader = resources::Reader::new(&catalog, 1024, 1024, |_, limit| {
        assert!(bytes.len() as u64 <= limit);
        reads.set(reads.get() + 1);
        Ok(Some(bytes.clone()))
    });
    assert!(matches!(
        reader.manifest(Digest::sha256(&bytes), 0),
        Err(AssessmentStoreError::ResourceLimit)
    ));
    assert_eq!(reads.get(), 1);
}
#[test]
fn scalar_chunks_retain_original_digest_validation_after_admission() {
    let bytes = b"ORIGINAL";
    let catalog = catalog(bytes);
    let mut reader =
        resources::Reader::new(&catalog, 1024, 1024, |_, _| Ok(Some(b"MODIFIED".to_vec())));
    assert!(matches!(
        reader.payload(&[chunk(bytes)], 8, Digest::sha256(bytes), 0),
        Err(AssessmentStoreError::Invalid(_))
    ));
}

#[test]
fn production_manifest_and_original_chunks_survive_admitted_cold_reads() {
    use runmat_analysis_fea::execution::{chunk_numerical_payload, AnalysisNumericalObjectKind};
    let original = b"ORIGINAL";
    let chunked = chunk_numerical_payload(
        AnalysisNumericalObjectKind::Observation,
        Digest::sha256(b"original-observation"),
        original,
        4096,
    )
    .unwrap();
    let manifest_bytes = chunked.manifest.canonical_bytes().unwrap();
    let root = catalog(&manifest_bytes).roots[0].clone();
    let child = catalog(original).roots[0].clone();
    let catalog =
        PublishedObjectCatalog::new(vec![root.clone()], vec![root.clone(), child.clone()]).unwrap();
    let reads = Cell::new(0);
    let mut reader = resources::Reader::new(&catalog, 1 << 20, 1 << 20, |digest, _| {
        reads.set(reads.get() + 1);
        Ok(if digest == root.logical_digest {
            Some(manifest_bytes.clone())
        } else if digest == child.logical_digest {
            Some(original.to_vec())
        } else {
            None
        })
    });
    let manifest = reader.manifest(root.logical_digest, 64 << 10).unwrap();
    assert_eq!(manifest, chunked.manifest);
    let retained = (64 << 10) + reader.resident(&manifest).unwrap();
    assert_eq!(
        reader
            .payload(
                &manifest.chunks,
                manifest.encoded_length,
                manifest.payload_digest,
                retained
            )
            .unwrap(),
        original
    );
    assert_eq!(reads.get(), 2);
}
