use std::collections::HashMap;

use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactResult, LogicalObject};
use runmat_meshing_core::{
    build_chunked_stage_payload, build_closed_stage_manifest, CanonicalMeshingContract,
    MeshingChunkMediaType, MeshingChunkPolicy, MeshingChunkStream, MeshingManifestDisposition,
    MeshingStageKind, MeshingStageResultKind, StableDigest,
};

use crate::{import_stage_objects, prepare_stage_objects, MeshingStageObjectRoot};

#[derive(Default)]
pub(crate) struct MemoryCache {
    objects: HashMap<Digest, Vec<u8>>,
}

impl MemoryCache {
    pub(crate) fn insert_all(&mut self, objects: &[LogicalObject]) {
        for object in objects {
            self.objects
                .insert(object.descriptor.digest, object.bytes.clone());
        }
    }
}

impl CacheImport for MemoryCache {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        Ok(self.objects.get(&digest).cloned())
    }
}

#[test]
fn stage_closure_round_trips_through_shared_objects() {
    let prepared = fixture(vec![vec![1; 200], vec![2; 300]], 1024);
    let mut cache = MemoryCache::default();
    cache.insert_all(&prepared.objects);

    let imported = import_stage_objects(
        &cache,
        prepared.root_reference(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();

    assert_eq!(imported, prepared);
    assert!(imported
        .objects
        .iter()
        .all(|object| object.descriptor.logical_name.starts_with("meshing/v2/")));
}

#[test]
fn large_payload_is_externalized_as_bounded_objects() {
    let prepared = fixture(vec![vec![7; 700_000], vec![8; 700_000]], 800_000);

    assert!(prepared.objects.len() >= 4);
    assert!(
        prepared
            .objects
            .iter()
            .map(|object| object.bytes.len())
            .sum::<usize>()
            > 1024 * 1024
    );
    assert!(prepared
        .objects
        .iter()
        .all(|object| object.bytes.len() <= 800_000));
}

#[test]
fn cache_claim_is_rehashed_before_decode() {
    let prepared = fixture(vec![vec![1; 200]], 1024);
    let mut cache = MemoryCache::default();
    cache.insert_all(&prepared.objects);
    cache
        .objects
        .insert(prepared.root.digest, b"poisoned".to_vec());

    let error = import_stage_objects(
        &cache,
        prepared.root_reference(),
        ObjectInventoryLimits::default(),
    )
    .unwrap_err();

    assert!(error.to_string().contains("wrong digest"));
}

#[test]
fn missing_chunk_rejects_interrupted_object_set() {
    let prepared = fixture(vec![vec![1; 200], vec![2; 300]], 512);
    let mut cache = MemoryCache::default();
    cache.insert_all(&prepared.objects);
    let chunk = prepared
        .objects
        .iter()
        .find(|object| object.descriptor.logical_name.contains("/chunks/"))
        .unwrap();
    cache.objects.remove(&chunk.descriptor.digest);

    let error = import_stage_objects(
        &cache,
        prepared.root_reference(),
        ObjectInventoryLimits::default(),
    )
    .unwrap_err();

    assert!(error.to_string().contains("unavailable"));
}

#[test]
fn independently_valid_root_must_still_close_over_result_identity() {
    let prepared = fixture(vec![vec![1; 200]], 1024);
    let mut cache = MemoryCache::default();
    cache.insert_all(&prepared.objects);
    let mut manifest = prepared.manifest.clone();
    manifest.invariant_summary_digest = digest(99);
    let bytes = manifest.canonical_encode().unwrap();
    let tampered_root = MeshingStageObjectRoot {
        digest: Digest::sha256(&bytes),
        encoded_length: bytes.len() as u64,
    };
    cache.objects.insert(tampered_root.digest, bytes);

    let error =
        import_stage_objects(&cache, tampered_root, ObjectInventoryLimits::default()).unwrap_err();

    assert!(error.to_string().contains("manifest closure"));
}

#[test]
fn object_inventory_limits_are_hard() {
    let (identity, manifest, chunks) = fixture_contracts(vec![vec![1; 200]], 1024);
    let error = prepare_stage_objects(
        identity,
        manifest,
        chunks,
        ObjectInventoryLimits {
            max_objects: 2,
            ..ObjectInventoryLimits::default()
        },
    )
    .unwrap_err();

    assert!(error.to_string().contains("too many"));
}

#[test]
fn import_enforces_total_bytes_before_accepting_closure() {
    let prepared = fixture(vec![vec![1; 200], vec![2; 300]], 512);
    let mut cache = MemoryCache::default();
    cache.insert_all(&prepared.objects);
    let error = import_stage_objects(
        &cache,
        prepared.root_reference(),
        ObjectInventoryLimits {
            max_total_bytes: prepared.root.encoded_length + 1,
            ..ObjectInventoryLimits::default()
        },
    )
    .unwrap_err();

    assert!(error.to_string().contains("inventory is too large"));
}

pub(crate) fn fixture(
    records: Vec<Vec<u8>>,
    maximum_chunk_bytes: u64,
) -> crate::PreparedMeshingStageObjects {
    let (identity, manifest, chunks) = fixture_contracts(records, maximum_chunk_bytes);
    prepare_stage_objects(identity, manifest, chunks, ObjectInventoryLimits::default()).unwrap()
}

fn fixture_contracts(
    records: Vec<Vec<u8>>,
    maximum_chunk_bytes: u64,
) -> (
    runmat_meshing_core::MeshingStageResultIdentity,
    runmat_meshing_core::MeshingStageManifest,
    Vec<runmat_meshing_core::EncodedMeshingChunk>,
) {
    let payload = build_chunked_stage_payload(
        &[MeshingChunkStream {
            media_type: MeshingChunkMediaType::MeshNodes,
            schema_version: 2,
            records,
        }],
        MeshingChunkPolicy {
            maximum_chunk_bytes,
            maximum_records_per_chunk: 10,
            maximum_total_encoded_bytes: 8 * 1024 * 1024,
        },
    )
    .unwrap();
    let (identity, manifest) = build_closed_stage_manifest(
        MeshingStageKind::Tetrahedralization,
        MeshingStageResultKind::WholeStage,
        digest(1),
        digest(2),
        Vec::new(),
        MeshingManifestDisposition::ValidatedDependency,
        &payload,
    )
    .unwrap();
    (identity, manifest, payload.chunks)
}

fn digest(seed: u8) -> StableDigest {
    StableDigest::from_bytes([seed; 32])
}
