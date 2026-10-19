//! Exact retained search differences against one immutable admitted state.
//! Connectivity and point insertion share quality encoding; relocation keeps its
//! own coordinate authority. Reconstruction never substitutes for admission.
use super::{
    error, DelaunayVolumeSliverError, DelaunayVolumeSliverErrorKind, DelaunayVolumeSliverOptions,
};
use crate::cdt::{
    topology::build_delaunay_volume_topology_with_regions, DelaunayTetrahedronQuality,
    DelaunayTopologyErrorKind, DelaunayVolumeNode, DelaunayVolumeQuality, DelaunayVolumeTopology,
};
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};
pub(super) mod connectivity;
use connectivity::connectivity_delta;
mod quality;
#[cfg(test)]
mod test_fixtures;
mod topology;
pub(super) use quality::QualityPatch;
pub(super) use topology::TopologyPatch;
fn identities(topology: &DelaunayVolumeTopology, vertices: [u32; 4]) -> [StableDigest; 4] {
    vertices.map(|index| topology.nodes[index as usize].identity)
}
