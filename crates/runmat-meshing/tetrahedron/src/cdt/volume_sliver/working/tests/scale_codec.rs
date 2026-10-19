//! Representation gate only. CAD/physics acceptance remains with the public codec.
use crate::cdt::DelaunayVolumeMesh;
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};

pub(super) fn round_trip(mesh: DelaunayVolumeMesh, expected_digest: [u8; 32]) {
    let limits = CanonicalLimits::new(1024 * 1024 * 1024, 50_000_000, 1024 * 1024, 64);
    let started = std::time::Instant::now();
    let bytes = encode_contract(b"scale-representation\0", "scale/v1", &mesh, limits).unwrap();
    eprintln!(
        "working-scale phase=encoded bytes={} elapsed_seconds={}",
        bytes.len(),
        started.elapsed().as_secs_f64()
    );
    // Cold read must not retain a second complete typed mesh.
    drop(mesh);
    let started = std::time::Instant::now();
    let decoded: DelaunayVolumeMesh =
        decode_contract(b"scale-representation\0", "scale/v1", &bytes, limits).unwrap();
    assert_eq!(super::scale_inventory::digest(&decoded), expected_digest);
    eprintln!(
        "working-scale phase=decoded elapsed_seconds={}",
        started.elapsed().as_secs_f64()
    );
}
