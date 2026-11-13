//! Deterministic conforming volume used by retained-owner and transport scale gates.
use crate::cdt::{
    topology::build_delaunay_volume_topology_with_regions, DelaunayTopologyOptions,
    DelaunayVolumeNode, DelaunayVolumeTopology,
};
use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::{NeverCancelled, StableDigest};
pub(in crate::cdt) fn region(name: &str) -> PersistentEntityId {
    PersistentEntityId::new(PersistentEntityKind::Region, name.into(), Vec::new())
}
pub(in crate::cdt) fn identity(value: u64) -> StableDigest {
    let mut bytes = [101; 32];
    bytes[..8].copy_from_slice(&value.to_be_bytes());
    StableDigest::from_bytes(bytes)
}
pub(in crate::cdt) fn grid(n: u32, thin: bool) -> DelaunayVolumeTopology {
    let width = n + 1;
    let slot = |x, y, z| (z * width + y) * width + x;
    let mut nodes = Vec::new();
    let depth = if thin { 1 } else { n };
    for z in 0..=depth {
        for y in 0..width {
            for x in 0..width {
                nodes.push(DelaunayVolumeNode::explicit(
                    identity(u64::from(slot(x, y, z))),
                    [f64::from(x), f64::from(y), f64::from(z)],
                ));
            }
        }
    }
    let region = Some(region("representative-working-region"));
    let mut cells = Vec::new();
    for z in 0..depth {
        for y in 0..n {
            for x in 0..n {
                let v = [
                    slot(x, y, z),
                    slot(x + 1, y, z),
                    slot(x, y + 1, z),
                    slot(x + 1, y + 1, z),
                    slot(x, y, z + 1),
                    slot(x + 1, y, z + 1),
                    slot(x, y + 1, z + 1),
                    slot(x + 1, y + 1, z + 1),
                ];
                for corners in [
                    [0, 1, 3, 7],
                    [0, 3, 2, 7],
                    [0, 2, 6, 7],
                    [0, 6, 4, 7],
                    [0, 4, 5, 7],
                    [0, 5, 1, 7],
                ] {
                    cells.push((corners.map(|i| v[i]), region.clone()));
                }
            }
        }
    }
    build_delaunay_volume_topology_with_regions(
        nodes,
        cells,
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}
