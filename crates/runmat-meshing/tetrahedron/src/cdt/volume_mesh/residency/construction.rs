//! Heap bytes of the intermediate results that volume construction holds before quality
//! convergence, and the forecast it reserves before each phase builds its working topology.
use super::topology_bytes;
use crate::cdt::mutation::TopologyMutation;
use crate::cdt::{
    DelaunayCarvedFacet, DelaunayCarving, DelaunayFacetRecovery, DelaunayFacetSteinerInsertion,
    DelaunayRecoveredFacet, DelaunayRecoveredFacetTriangle, DelaunayRecoveredSegment,
    DelaunayRecoveredSegmentNode, DelaunaySegmentRecovery, DelaunayVolumeTopology,
};
use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::resident_bytes::vec_bytes;
use runmat_meshing_core::StableDigest;

/// Growth window of a phase's working topology over the inventory it starts from, as in quality
/// convergence: one eighth more nodes and tetrahedra.
const WINDOW_DIVISOR: usize = 8;

/// A Delaunay tetrahedralization of a point set holds about six and a half tetrahedra per point;
/// the point-set phase forecasts eight before it knows its tetrahedra and re-measures after.
const POINT_SET_TETRAHEDRA_PER_NODE: usize = 8;

/// Forecast of the working topology a phase builds from `nodes` and `tetrahedra`, including its
/// growth window.
pub(in crate::cdt) fn working_forecast(nodes: usize, tetrahedra: usize) -> u64 {
    TopologyMutation::forecast_bytes(
        nodes.saturating_add(nodes / WINDOW_DIVISOR + 1),
        tetrahedra.saturating_add(tetrahedra / WINDOW_DIVISOR + 1),
    )
}

/// Forecast of the point-set phase's working topology over `nodes` constraint nodes.
pub(in crate::cdt) fn point_set_forecast(nodes: usize) -> u64 {
    working_forecast(nodes, nodes.saturating_mul(POINT_SET_TETRAHEDRA_PER_NODE))
}

/// The working forecast of a phase that rebuilds or edits `topology`.
pub(in crate::cdt) fn rebuild_forecast(topology: &DelaunayVolumeTopology) -> u64 {
    working_forecast(topology.nodes.len(), topology.tetrahedra.len())
}

fn recovered_segments_bytes(segments: &Vec<DelaunayRecoveredSegment>) -> u64 {
    segments.iter().fold(
        vec_bytes::<DelaunayRecoveredSegment>(segments.capacity()),
        |sum, segment| {
            sum.saturating_add(vec_bytes::<DelaunayRecoveredSegmentNode>(
                segment.nodes.capacity(),
            ))
        },
    )
}

/// Heap bytes of a segment recovery: its topology, recovered chains and Steiner identities.
pub(in crate::cdt) fn segment_recovery_bytes(recovery: &DelaunaySegmentRecovery) -> u64 {
    topology_bytes(&recovery.topology)
        .saturating_add(recovered_segments_bytes(&recovery.segments))
        .saturating_add(vec_bytes::<StableDigest>(
            recovery.steiner_node_identities.capacity(),
        ))
}

/// Heap bytes of a facet recovery, including the segment recovery it keeps as its prerequisite.
pub(in crate::cdt) fn facet_recovery_bytes(recovery: &DelaunayFacetRecovery) -> u64 {
    let facets = recovery.facets.iter().fold(
        vec_bytes::<DelaunayRecoveredFacet>(recovery.facets.capacity()),
        |sum, facet| {
            sum.saturating_add(vec_bytes::<DelaunayRecoveredFacetTriangle>(
                facet.triangles.capacity(),
            ))
        },
    );
    [
        segment_recovery_bytes(&recovery.segment_recovery),
        recovered_segments_bytes(&recovery.recovered_segments),
        topology_bytes(&recovery.topology),
        facets,
        vec_bytes::<DelaunayFacetSteinerInsertion>(recovery.steiner_insertions.capacity()),
    ]
    .into_iter()
    .fold(0, u64::saturating_add)
}

/// Heap bytes of a carved volume: its topology, removed cells and facet sides.
pub(in crate::cdt) fn carving_bytes(carving: &DelaunayCarving) -> u64 {
    carving.facets.iter().fold(
        topology_bytes(&carving.topology)
            .saturating_add(vec_bytes::<[StableDigest; 4]>(
                carving.removed_tetrahedra.capacity(),
            ))
            .saturating_add(vec_bytes::<DelaunayCarvedFacet>(carving.facets.capacity())),
        |sum, facet| {
            sum.saturating_add(vec_bytes::<PersistentEntityId>(facet.region_ids.capacity()))
        },
    )
}

/// Forecast of what assembling several completed components builds beside them: the merged
/// node map, the identity-keyed cell list (counted twice while it grows), the node index, and
/// the working topology over every node and tetrahedron. The assembled mesh's quality is
/// charged by the caller once it exists.
pub(in crate::cdt) fn assembly_forecast(meshes: &[crate::cdt::DelaunayVolumeMesh]) -> u64 {
    use runmat_meshing_core::resident_bytes::btree_bytes;
    let (nodes, tetrahedra) = meshes
        .iter()
        .fold((0_usize, 0_usize), |(nodes, cells), mesh| {
            (
                nodes.saturating_add(mesh.topology.nodes.len()),
                cells.saturating_add(mesh.topology.tetrahedra.len()),
            )
        });
    [
        btree_bytes::<StableDigest, crate::cdt::DelaunayVolumeNode>(nodes),
        vec_bytes::<([StableDigest; 4], Option<PersistentEntityId>)>(tetrahedra).saturating_mul(2),
        btree_bytes::<StableDigest, u32>(nodes),
        working_forecast(nodes, tetrahedra),
    ]
    .into_iter()
    .fold(0, u64::saturating_add)
}
