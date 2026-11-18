//! Heap bytes that independent volume validation holds beside the mesh it checks.
use crate::cdt::{DelaunayConstraintFacet, DelaunayConstraintNode, DelaunayConstraintSegment};
use crate::cdt::{DelaunayConstraints, DelaunayVolumeMesh};
use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::resident_bytes::{btree_bytes, btree_family_bytes, vec_bytes};
use runmat_meshing_core::StableDigest;

/// Heap bytes of a constraint inventory's vectors. Shared persistent identities belong to the
/// boundary model and are not counted.
pub(in crate::cdt) fn constraints_bytes(constraints: &DelaunayConstraints) -> u64 {
    constraints.facets.iter().fold(
        vec_bytes::<DelaunayConstraintNode>(constraints.nodes.capacity())
            .saturating_add(vec_bytes::<DelaunayConstraintSegment>(
                constraints.segments.capacity(),
            ))
            .saturating_add(vec_bytes::<DelaunayConstraintFacet>(
                constraints.facets.capacity(),
            )),
        |sum, facet| {
            sum.saturating_add(vec_bytes::<PersistentEntityId>(
                facet.contact_ids.capacity(),
            ))
        },
    )
}

/// Upper bound on the scratch that `validate_delaunay_volume_mesh` allocates beside `mesh` and
/// the `constraints` it rebuilds, counted as if every check's scratch were live at once:
///
/// - the protected-face list and its ordered set, one entry per facet binding;
/// - the canonical face-use table, four entries per tetrahedron, and one cursor per region;
/// - node-identity sets over every topology node and refinement insertion (provenance and
///   mutation lineage keep two at a time);
/// - the exact-source maps: admitted vertices, exact edges, region sides, and one chart set per
///   source face or contact identity;
/// - quality validation's boundary-node, protected-facet and protected-edge sets.
///
/// Scoped grading also derives one metric context per tetrahedron; `scoped_metric` adds those
/// records and their resolved metrics, excluding the identity vectors they share with
/// provenance.
pub(in crate::cdt) fn validation_scratch_bytes(
    mesh: &DelaunayVolumeMesh,
    constraints: &DelaunayConstraints,
    exact_edges: usize,
    scoped_metric: bool,
) -> u64 {
    let facets = mesh.provenance.facets.len();
    let tetrahedra = mesh.topology.tetrahedra.len();
    let lineage = mesh
        .topology
        .nodes
        .len()
        .saturating_add(mesh.quality_mutations.len());
    let chart_owners = constraints
        .facets
        .iter()
        .fold(constraints.facets.len(), |sum, facet| {
            sum.saturating_add(facet.contact_ids.len())
        });
    let scoped = if scoped_metric {
        vec_bytes::<crate::cdt::volume_quality::context::DelaunayVolumeMetricContext>(tetrahedra)
            .saturating_add(vec_bytes::<
                runmat_meshing_size::metric::ResolvedMetricEvaluation,
            >(tetrahedra))
    } else {
        0
    };
    [
        vec_bytes::<[StableDigest; 3]>(facets),
        btree_bytes::<[StableDigest; 3], ()>(facets),
        vec_bytes::<([u32; 3], usize, usize)>(tetrahedra.saturating_mul(4)),
        vec_bytes::<usize>(mesh.topology.incidence.regions.len()),
        btree_bytes::<StableDigest, ()>(lineage).saturating_mul(2),
        btree_bytes::<PersistentEntityId, StableDigest>(constraints.nodes.len()),
        btree_bytes::<PersistentEntityId, ()>(exact_edges),
        btree_bytes::<PersistentEntityId, ()>(constraints.facets.len().saturating_mul(2)),
        btree_family_bytes::<PersistentEntityId, std::collections::BTreeSet<StableDigest>>(
            1,
            chart_owners,
        ),
        btree_family_bytes::<StableDigest, ()>(chart_owners, chart_owners),
        btree_bytes::<StableDigest, ()>(facets.saturating_mul(3)),
        btree_bytes::<[StableDigest; 3], ()>(facets),
        btree_bytes::<[StableDigest; 2], ()>(facets.saturating_mul(3)),
        scoped,
    ]
    .into_iter()
    .fold(0, u64::saturating_add)
}
