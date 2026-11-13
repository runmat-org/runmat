use super::*;
use crate::{serial_tests::request, NoopMeshingProgress};
use runmat_meshing_core::{
    MeshingCancellationSignal, MeshingChunkMediaType, MeshingFailureCategory, MeshingStageKind,
    NeverCancelled,
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn streamed_publication_digest_matches_original_framing() {
    let ceilings = crate::serial_tests::ceilings();
    let streams = [
        MeshingChunkStream {
            media_type: MeshingChunkMediaType::SurfacePartitions,
            schema_version: 7,
            records: vec![vec![3; 2 * 1024 * 1024 + 7], vec![]],
        },
        MeshingChunkStream {
            media_type: MeshingChunkMediaType::SurfaceMesh,
            schema_version: 2,
            records: vec![vec![1, 2, 3]],
        },
    ];
    let domain = b"runmat-exact-surface-publication-validation/v1\0";
    let mut original = domain.to_vec();
    for stream in &streams {
        original.extend_from_slice(stream.media_type.media_type().as_bytes());
        original.extend_from_slice(&stream.schema_version.to_be_bytes());
        for record in &stream.records {
            original.extend_from_slice(&(record.len() as u64).to_be_bytes());
            original.extend_from_slice(record);
        }
    }
    let request = request();
    let mut progress = NoopMeshingProgress;
    let mut control = MeshingStageControl::new(
        MeshingStageKind::SurfaceMesh,
        0,
        &request,
        ceilings,
        &NeverCancelled,
        &mut progress,
    )
    .unwrap();
    let actual = super::streams(
        domain,
        &streams,
        &mut control,
        &MeshingStageCheckpoint::default(),
    )
    .unwrap();
    assert_eq!(actual.bytes(), Digest::sha256(original).bytes());
    let actual = record(
        b"record\0",
        &streams[0].records[0],
        &mut control,
        &MeshingStageCheckpoint::default(),
    )
    .unwrap();
    assert_eq!(
        actual.bytes(),
        Digest::sha256_parts([b"record\0".as_slice(), &streams[0].records[0]]).bytes()
    );
}

#[test]
fn publication_hash_preserves_one_shot_typed_cancellation() {
    struct Pulse(AtomicUsize);
    impl MeshingCancellationSignal for Pulse {
        fn is_cancelled(&self) -> bool {
            self.0.fetch_add(1, Ordering::Relaxed) == 3
        }
    }
    let request = request();
    let ceilings = crate::serial_tests::ceilings();
    let pulse = Pulse(AtomicUsize::new(0));
    let mut progress = NoopMeshingProgress;
    let mut control = MeshingStageControl::new(
        MeshingStageKind::SurfaceMesh,
        0,
        &request,
        ceilings,
        &pulse,
        &mut progress,
    )
    .unwrap();
    let failure = record(
        b"record\0",
        &vec![1; 4 * 1024 * 1024],
        &mut control,
        &MeshingStageCheckpoint::default(),
    )
    .unwrap_err();
    assert_eq!(failure.category, MeshingFailureCategory::Cancelled);
    assert_eq!(pulse.0.load(Ordering::Relaxed), 4);
}
