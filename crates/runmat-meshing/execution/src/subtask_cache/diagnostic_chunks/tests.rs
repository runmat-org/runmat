use super::*;
use crate::subtask_cache::record_chunks::{self, Manifest, CHUNK, LIMITS, MAXIMUM};
use runmat_geometry_core::GeometryAllocationReservation;
const PREFIX: &[u8] = DIAGNOSTIC.prefix;
fn retain(
    store: &dyn MeshingSubtaskCacheStore,
    bytes: &[u8],
    materialization: u64,
    control: &dyn GeometryEvaluationControl,
) -> Result<Option<StableDigest>, MeshingSubtaskCacheError> {
    record_chunks::retain(DIAGNOSTIC, store, bytes, materialization, control)
}
fn read_volume_diagnostic<'a>(
    store: &dyn MeshingSubtaskCacheStore,
    key: StableDigest,
    control: &'a dyn GeometryEvaluationControl,
) -> Result<Option<MeshingSubtaskCacheEntry<'a>>, MeshingSubtaskCacheError> {
    let codec = crate::codec_control::CodecControl::new(control);
    record_chunks::read(DIAGNOSTIC, store, key, control, |bytes| {
        runmat_canonical_codec::canonical_chunked_payload_materialization_resident_bytes_estimate(
            [bytes],
            &codec,
        )
        .map_err(|error| codec.error(error))
    })
}
fn digest(
    bytes: &[u8],
    control: &dyn GeometryEvaluationControl,
) -> Result<StableDigest, MeshingSubtaskCacheError> {
    record_chunks::digest(DIAGNOSTIC, bytes, control)
}
fn manifest_digest(
    bytes: &[u8],
    control: &dyn GeometryEvaluationControl,
) -> Result<StableDigest, MeshingSubtaskCacheError> {
    record_chunks::manifest_digest(DIAGNOSTIC, bytes, control)
}

use runmat_geometry_core::{GeometryEvaluationError, GeometryEvaluationErrorKind};
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    },
};

struct Control {
    live: AtomicU64,
    maximum: u64,
}
impl Control {
    fn new(maximum: u64) -> Self {
        Self {
            live: AtomicU64::new(0),
            maximum,
        }
    }
}
impl GeometryEvaluationControl for Control {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_allocation_bytes(&self, bytes: u64) -> Result<(), GeometryEvaluationError> {
        self.live
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
                live.checked_add(bytes).filter(|next| *next <= self.maximum)
            })
            .map(|_| ())
            .map_err(|_| {
                GeometryEvaluationError::new(
                    GeometryEvaluationErrorKind::AllocationBudgetExceeded,
                    "diagnostic test ceiling",
                )
            })
    }
    fn release_allocation_bytes(&self, bytes: u64) {
        self.live.fetch_sub(bytes, Ordering::Relaxed);
    }
    fn consume_search_work(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
}
#[derive(Debug, Default)]
struct Store(Mutex<BTreeMap<StableDigest, (Vec<u8>, u64)>>);
impl MeshingSubtaskCacheStore for Store {
    fn implementation_identity(&self) -> StableDigest {
        StableDigest::from_bytes([1; 32])
    }
    fn maximum_entry_bytes(&self) -> u64 {
        8 * 1024 * 1024
    }
    fn read<'a>(
        &self,
        key: StableDigest,
        control: &'a dyn GeometryEvaluationControl,
    ) -> Result<Option<MeshingSubtaskCacheEntry<'a>>, MeshingSubtaskCacheError> {
        let records = self.0.lock().unwrap();
        let Some((bytes, workspace)) = records.get(&key) else {
            return Ok(None);
        };
        let storage = GeometryAllocationReservation::new(control, bytes.len() as u64 + workspace)
            .map_err(MeshingSubtaskCacheError::Control)?;
        Ok(Some(MeshingSubtaskCacheEntry {
            bytes: bytes.clone(),
            storage,
            materialization_resident_bytes: *workspace,
        }))
    }
    fn write(
        &self,
        key: StableDigest,
        bytes: &[u8],
        workspace: u64,
        _: &dyn GeometryEvaluationControl,
    ) -> Result<bool, MeshingSubtaskCacheError> {
        self.0
            .lock()
            .unwrap()
            .insert(key, (bytes.to_vec(), workspace));
        Ok(true)
    }
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Manifest-last chunk transport reassembles exact bytes, rejects corruption and
//   understated storage, treats missing chunks as a miss, and releases admitted storage after
//   failure or record drop. This fixture does not certify every native store eviction schedule or
//   whole-process memory.
#[test]
fn multi_chunk_diagnostic_round_trip_corruption_miss_and_ceiling_release_storage() {
    let store = Store::default();
    let control = Control::new(64 * 1024 * 1024);
    let codec = crate::codec_control::CodecControl::new(&control);
    let bytes = runmat_canonical_codec::encode_contract_with_control(
        b"diagnostic-test/v1\0",
        "diagnostic-test/v1",
        &"x".repeat(CHUNK + 17),
        runmat_canonical_codec::CanonicalLimits::new(8 * 1024 * 1024, 4096, 8 * 1024 * 1024, 16),
        &codec,
    )
    .unwrap();
    let materialization =
        runmat_canonical_codec::canonical_chunked_payload_materialization_resident_bytes_estimate(
            std::iter::once(bytes.as_slice()),
            &codec,
        )
        .unwrap();
    let key = retain(&store, &bytes, materialization, &control)
        .unwrap()
        .unwrap();
    let record = read_volume_diagnostic(&store, key, &control)
        .unwrap()
        .unwrap();
    assert_eq!(record.bytes, bytes);
    assert_eq!(
        control.live.load(Ordering::Relaxed),
        bytes.len() as u64 + materialization
    );
    drop(record);
    assert_eq!(control.live.load(Ordering::Relaxed), 0);
    let limited = Control::new(512 * 1024);
    assert!(read_volume_diagnostic(&store, key, &limited).is_err());
    assert_eq!(limited.live.load(Ordering::Relaxed), 0);
    let mut manifest: Manifest = runmat_canonical_codec::decode_contract_with_control(
        PREFIX,
        "meshing-diagnostic-chunks/v1",
        &store.0.lock().unwrap()[&key].0,
        LIMITS,
        &codec,
    )
    .unwrap();
    manifest.materialization = 0;
    let forged = runmat_canonical_codec::encode_contract_with_control(
        PREFIX,
        "meshing-diagnostic-chunks/v1",
        &manifest,
        LIMITS,
        &codec,
    )
    .unwrap();
    let forged_key = manifest_digest(&forged, &control).unwrap();
    store
        .write(forged_key, &forged, 256 * 1024, &control)
        .unwrap();
    assert!(read_volume_diagnostic(&store, forged_key, &control).is_err());
    assert_eq!(control.live.load(Ordering::Relaxed), 0);
    let first = digest(&bytes[..CHUNK], &control).unwrap();
    store.0.lock().unwrap().get_mut(&first).unwrap().0[0] ^= 1;
    assert!(read_volume_diagnostic(&store, key, &control).is_err());
    assert_eq!(control.live.load(Ordering::Relaxed), 0);
    store.0.lock().unwrap().remove(&first);
    assert!(read_volume_diagnostic(&store, key, &control)
        .unwrap()
        .is_none());
    assert_eq!(control.live.load(Ordering::Relaxed), 0);
}

#[test]
fn manifest_identity_and_total_inventory_are_checked_before_payload_allocation() {
    let store = Store::default();
    let control = Control::new(512 * 1024);
    let codec = crate::codec_control::CodecControl::new(&control);
    let bytes = runmat_canonical_codec::encode_contract_with_control(
        PREFIX,
        "meshing-diagnostic-chunks/v1",
        &Manifest {
            length: MAXIMUM + 1,
            chunk_size: CHUNK,
            materialization: 0,
            chunks: Vec::new(),
        },
        LIMITS,
        &codec,
    )
    .unwrap();
    let key = manifest_digest(&bytes, &control).unwrap();
    store.write(key, &bytes, 256 * 1024, &control).unwrap();
    assert!(read_volume_diagnostic(&store, key, &control).is_err());
    assert_eq!(control.live.load(Ordering::Relaxed), 0);
    store.0.lock().unwrap().get_mut(&key).unwrap().0[0] ^= 1;
    assert!(read_volume_diagnostic(&store, key, &control).is_err());
    assert_eq!(control.live.load(Ordering::Relaxed), 0);
}

#[test]
fn maximum_chunk_manifest_fits_the_existing_bounded_codec_without_payload_allocation() {
    let count = MAXIMUM / CHUNK;
    let manifest = Manifest {
        length: MAXIMUM,
        chunk_size: CHUNK,
        materialization: 1,
        chunks: (0..count)
            .map(|index| {
                let mut bytes = [0; 32];
                bytes[..8].copy_from_slice(&(index as u64).to_be_bytes());
                StableDigest::from_bytes(bytes)
            })
            .collect(),
    };
    let bytes = runmat_canonical_codec::encode_contract(
        PREFIX,
        "meshing-diagnostic-chunks/v1",
        &manifest,
        LIMITS,
    )
    .unwrap();
    let decoded: Manifest = runmat_canonical_codec::decode_contract(
        PREFIX,
        "meshing-diagnostic-chunks/v1",
        &bytes,
        LIMITS,
    )
    .unwrap();
    assert_eq!(decoded.chunks, manifest.chunks);
    assert_eq!(decoded.length, MAXIMUM);
    assert_eq!(count, 256);
}
