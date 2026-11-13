//! A deterministic work witness for dense, exactly coplanar boundary samples.
use super::*;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Default)]
struct CountingProbe(AtomicUsize);
impl MeshingCancellationSignal for CountingProbe {
    fn is_cancelled(&self) -> bool {
        self.0.fetch_add(1, Ordering::Relaxed);
        false
    }
}

#[test]
fn dense_boundary_bootstrap_uses_local_work_and_retains_every_sample() {
    let mut nodes = Vec::new();
    for x in 0..12 {
        for y in 0..12 {
            for z in 0..12 {
                if [x, y, z].iter().all(|value| *value != 0 && *value != 11) {
                    continue;
                }
                let identity = StableDigest::from_bytes(
                    Sha256::digest(format!("dense-boundary/{x}/{y}/{z}").as_bytes()).into(),
                );
                nodes.push(DelaunayVolumeNode::explicit(
                    identity,
                    [x as f64, y as f64, z as f64],
                ));
            }
        }
    }
    let expected = nodes.len();
    let probe = CountingProbe::default();
    let mut options = DelaunayPointSetOptions::default();
    options.insertion.topology.cancellation_check_interval = 1;
    let topology = build_delaunay_volume_point_set(nodes, options, &probe).unwrap();
    assert_eq!(topology.nodes.len(), expected);
    assert!(topology
        .incidence
        .vertex_stars
        .iter()
        .all(|star| !star.is_empty()));
    let checkpoints = probe.0.load(Ordering::Relaxed);
    assert!(
        checkpoints < 500_000,
        "dense boundary required {checkpoints} work checkpoints"
    );
}
