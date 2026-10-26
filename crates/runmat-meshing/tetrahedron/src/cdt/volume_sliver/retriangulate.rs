use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    quality::predicate::{
        point_in_closed_triangle_surface, tetrahedron_centroid, tetrahedron_edge_aspect_ratio,
        tetrahedron_scaled_jacobian, tetrahedron_volume, PointInClosedSurface,
    },
    quality::tolerance::MeshingTolerance,
    MeshingCancellationSignal,
};

use super::{error, DelaunayVolumeSliverError, DelaunayVolumeSliverErrorKind};
use crate::{
    cavity::constrained::{
        constrained_cavity_boundary_node_ids, constrained_cavity_from_selected_tetrahedra,
        constrained_cavity_star_refill_for_apex, CavityTetrahedron,
        ConstrainedCavityExtractionError,
    },
    cdt::{
        insertion::{
            insert_delaunay_volume_node_locally_with_barriers,
            legalize_delaunay_volume_topology_with_barriers,
        },
        topology::build_delaunay_volume_topology_with_regions,
        DelaunayInsertionErrorKind, DelaunayPointSetErrorKind, DelaunayPointSetOptions,
        DelaunayTopologyErrorKind, DelaunayVolumeNode, DelaunayVolumeTopology,
    },
};

mod preparation;

pub(super) struct PreparedVertexRemoval<'a> {
    topology: &'a DelaunayVolumeTopology,
    source_index: usize,
    insertion_options: crate::cdt::DelaunayInsertionOptions,
    selected: BTreeSet<usize>,
    region: runmat_geometry_core::PersistentEntityId,
    refill_tetrahedra: Vec<[u32; 4]>,
}
impl<'a> PreparedVertexRemoval<'a> {
    pub(super) fn prepare(
        topology: &'a DelaunayVolumeTopology,
        source_index: usize,
        insertion_options: crate::cdt::DelaunayInsertionOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Option<Self>, DelaunayVolumeSliverError> {
        preparation::prepare(topology, source_index, insertion_options, cancellation)
    }
    pub(super) fn relocate(
        &self,
        replacement: DelaunayVolumeNode,
        protected_faces: &[[runmat_meshing_core::StableDigest; 3]],
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Option<DelaunayVolumeTopology>, DelaunayVolumeSliverError> {
        let insertion_options = self.insertion_options;
        if self.topology.nodes.iter().enumerate().any(|(index, node)| {
            index != self.source_index
                && (node.identity == replacement.identity
                    || node.coordinates_m.canonical_value()
                        == replacement.coordinates_m.canonical_value())
        }) {
            return Ok(None);
        }
        let Some(topology) = self.materialize(protected_faces, cancellation)? else {
            return Ok(None);
        };
        match insert_delaunay_volume_node_locally_with_barriers(
            topology,
            replacement,
            protected_faces,
            insertion_options,
            cancellation,
        ) {
            Ok(topology) => Ok(Some(topology)),
            Err(failure) => insertion_result(failure.kind, failure.to_string()),
        }
    }
    pub(super) fn materialize(
        &self,
        protected_faces: &[[runmat_meshing_core::StableDigest; 3]],
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Option<DelaunayVolumeTopology>, DelaunayVolumeSliverError> {
        let insertion_options = self.insertion_options;
        let topology = self.topology;
        let source_index = self.source_index;
        let region = &self.region;
        let refill_tetrahedra = &self.refill_tetrahedra;
        let source_identity = topology.nodes[source_index].identity;
        let nodes = topology
            .nodes
            .iter()
            .copied()
            .filter(|node| node.identity != source_identity)
            .collect::<Vec<_>>();
        // Canonical nodes are ordered by unique identity. Removing one entry
        // preserves that order, so its index shift is exact and needs no map.
        let remap = |vertices| remap_without_vertex(vertices, source_index);
        let removed = &self.selected;
        let mut tetrahedra = topology
            .tetrahedra
            .iter()
            .enumerate()
            .filter(|(index, _)| !removed.contains(index))
            .map(|(_, tetrahedron)| {
                Ok((
                    remap(tetrahedron.vertex_indices)?,
                    tetrahedron.region_id.clone(),
                ))
            })
            .collect::<Result<Vec<_>, DelaunayVolumeSliverError>>()?;
        for tetrahedron in refill_tetrahedra {
            let remapped = remap(*tetrahedron)?;
            tetrahedra.push((remapped, Some(region.clone())));
        }
        let rebuilt = match build_delaunay_volume_topology_with_regions(
            nodes,
            tetrahedra,
            insertion_options.topology,
            cancellation,
        ) {
            Ok(topology) => topology,
            Err(failure) => return topology_result(failure.kind, failure.to_string()),
        };
        match legalize_delaunay_volume_topology_with_barriers(
            rebuilt,
            protected_faces,
            insertion_options,
            cancellation,
        ) {
            Ok(topology) => Ok(Some(topology)),
            Err(failure) => insertion_result(failure.kind, failure.to_string()),
        }
    }
}

fn remap_without_vertex(
    vertices: [u32; 4],
    source_index: usize,
) -> Result<[u32; 4], DelaunayVolumeSliverError> {
    let remap = |vertex: u32| {
        let index = vertex as usize;
        if index == source_index {
            return Err(error(
                DelaunayVolumeSliverErrorKind::InvalidTopology,
                "interior vertex-star cavity remap referenced the removed node",
            ));
        }
        Ok(vertex - u32::from(index > source_index))
    };
    Ok([
        remap(vertices[0])?,
        remap(vertices[1])?,
        remap(vertices[2])?,
        remap(vertices[3])?,
    ])
}

/// Most interior vertex stars are convex. For those, the canonical Delaunay fill of the star is
/// the point-set tetrahedralization of its boundary vertices. Exact boundary equality is the
/// admission proof; a non-convex result is not a legal candidate for this bounded relocation.
fn convex_delaunay_refill(
    topology: &DelaunayVolumeTopology,
    cavity: &crate::cavity::constrained::ConstrainedCavity,
    boundary_node_ids: &BTreeSet<u32>,
    insertion_options: crate::cdt::DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<Vec<[u32; 4]>>, DelaunayVolumeSliverError> {
    let nodes = boundary_node_ids
        .iter()
        .map(|index| topology.nodes[*index as usize])
        .collect::<Vec<_>>();
    let local = match crate::cdt::build_delaunay_volume_point_set(
        nodes,
        DelaunayPointSetOptions {
            insertion: insertion_options,
        },
        cancellation,
    ) {
        Ok(topology) => topology,
        Err(failure) => match failure.kind {
            DelaunayPointSetErrorKind::ResourceLimit => {
                return Err(error(
                    DelaunayVolumeSliverErrorKind::ResourceLimit,
                    failure.to_string(),
                ))
            }
            DelaunayPointSetErrorKind::Cancelled => {
                return Err(error(
                    DelaunayVolumeSliverErrorKind::Cancelled,
                    failure.to_string(),
                ))
            }
            _ => return Ok(None),
        },
    };
    // Only the deleted star's boundary participates in its local point set.
    // Do not rebuild an identity map for every unrelated node in the volume.
    let old_indices = boundary_node_ids
        .iter()
        .map(|index| (topology.nodes[*index as usize].identity, *index))
        .collect::<BTreeMap<_, _>>();
    let expected_boundary = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let boundary_triangles = cavity
        .boundary_faces
        .iter()
        .map(|face| {
            face.node_ids.map(|index| {
                topology.nodes[index as usize]
                    .coordinates_m
                    .canonical_value()
            })
        })
        .collect::<Vec<_>>();
    let admitted = local
        .tetrahedra
        .iter()
        .filter(|tetrahedron| {
            let points = tetrahedron
                .vertex_indices
                .map(|index| local.nodes[index as usize].coordinates_m.canonical_value());
            point_in_closed_triangle_surface(
                tetrahedron_centroid(points),
                &boundary_triangles,
                MeshingTolerance::default(),
            ) == PointInClosedSurface::Inside
        })
        .map(|tetrahedron| {
            tetrahedron
                .vertex_indices
                .map(|index| old_indices[&local.nodes[index as usize].identity])
        })
        .collect::<Vec<_>>();
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for tetrahedron in &admitted {
        for omitted in 0..4 {
            let face = std::array::from_fn(|slot| tetrahedron[(omitted + slot + 1) % 4]);
            *face_counts.entry(sorted_face(face)).or_default() += 1;
        }
    }
    let observed_boundary = face_counts
        .into_iter()
        .filter_map(|(face, count)| (count == 1).then_some(face))
        .collect::<BTreeSet<_>>();
    if observed_boundary != expected_boundary {
        return Ok(None);
    }
    Ok(Some(admitted))
}

fn sorted_face(mut face: [u32; 3]) -> [u32; 3] {
    face.sort_unstable();
    face
}

fn topology_result(
    kind: DelaunayTopologyErrorKind,
    reason: String,
) -> Result<Option<DelaunayVolumeTopology>, DelaunayVolumeSliverError> {
    match kind {
        DelaunayTopologyErrorKind::ResourceLimit => {
            Err(error(DelaunayVolumeSliverErrorKind::ResourceLimit, reason))
        }
        DelaunayTopologyErrorKind::Cancelled => {
            Err(error(DelaunayVolumeSliverErrorKind::Cancelled, reason))
        }
        _ => Ok(None),
    }
}

fn insertion_result(
    kind: DelaunayInsertionErrorKind,
    reason: String,
) -> Result<Option<DelaunayVolumeTopology>, DelaunayVolumeSliverError> {
    match kind {
        DelaunayInsertionErrorKind::ResourceLimit => {
            Err(error(DelaunayVolumeSliverErrorKind::ResourceLimit, reason))
        }
        DelaunayInsertionErrorKind::Cancelled => {
            Err(error(DelaunayVolumeSliverErrorKind::Cancelled, reason))
        }
        _ => Ok(None),
    }
}
