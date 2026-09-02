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

/// Replaces an interior vertex by deleting its complete star, filling the resulting closed
/// cavity without that vertex, and inserting the replacement through protected Bowyer-Watson.
/// Connectivity is therefore rebuilt around the moved point instead of being frozen in place.
pub(super) fn retriangulate_relocated_vertex(
    topology: &DelaunayVolumeTopology,
    source_index: usize,
    replacement: DelaunayVolumeNode,
    protected_faces: &[[runmat_meshing_core::StableDigest; 3]],
    insertion_options: crate::cdt::DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<DelaunayVolumeTopology>, DelaunayVolumeSliverError> {
    let Some(topology) = retriangulate_without_vertex(
        topology,
        source_index,
        Some(replacement),
        protected_faces,
        insertion_options,
        cancellation,
    )?
    else {
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

pub(super) fn retriangulate_deleted_vertex(
    topology: &DelaunayVolumeTopology,
    source_index: usize,
    protected_faces: &[[runmat_meshing_core::StableDigest; 3]],
    insertion_options: crate::cdt::DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<DelaunayVolumeTopology>, DelaunayVolumeSliverError> {
    retriangulate_without_vertex(
        topology,
        source_index,
        None,
        protected_faces,
        insertion_options,
        cancellation,
    )
}

fn retriangulate_without_vertex(
    topology: &DelaunayVolumeTopology,
    source_index: usize,
    replacement: Option<DelaunayVolumeNode>,
    protected_faces: &[[runmat_meshing_core::StableDigest; 3]],
    insertion_options: crate::cdt::DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<DelaunayVolumeTopology>, DelaunayVolumeSliverError> {
    if source_index >= topology.nodes.len()
        || topology.nodes.iter().enumerate().any(|(index, node)| {
            index != source_index
                && replacement.is_some_and(|replacement| {
                    node.identity == replacement.identity
                        || node.coordinates_m == replacement.coordinates_m
                })
        })
    {
        return Ok(None);
    }

    let selected = topology.incidence.vertex_stars[source_index]
        .iter()
        .map(|index| *index as usize)
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Ok(None);
    }
    let regions = selected
        .iter()
        .filter_map(|index| topology.tetrahedra[*index].region_id.clone())
        .collect::<BTreeSet<_>>();
    if regions.len() != 1 {
        return Ok(None);
    }
    let region = regions
        .into_iter()
        .next()
        .expect("one region was established");

    let cavity_tetrahedra = topology
        .tetrahedra
        .iter()
        .enumerate()
        .map(|(index, tetrahedron)| {
            let points = tetrahedron
                .vertex_indices
                .map(|vertex| topology.nodes[vertex as usize].coordinates_m);
            CavityTetrahedron {
                tetrahedron_id: index as u32,
                component_id: 0,
                node_ids: tetrahedron.vertex_indices,
                source_surface_element_id: 0,
                region_ids: tetrahedron
                    .region_id
                    .iter()
                    .map(|region| region.source_topology_id.clone())
                    .collect(),
                volume_m3: tetrahedron_volume(points),
                aspect_ratio: tetrahedron_edge_aspect_ratio(points),
                exact_scaled_jacobian: tetrahedron_scaled_jacobian(points),
            }
        })
        .collect::<Vec<_>>();
    let cavity = match constrained_cavity_from_selected_tetrahedra(
        &cavity_tetrahedra,
        &selected,
        Vec::new(),
    ) {
        Ok(cavity) => cavity,
        Err(ConstrainedCavityExtractionError::Validation(_)) => return Ok(None),
        Err(failure) => {
            return Err(error(
                DelaunayVolumeSliverErrorKind::InvalidTopology,
                format!("interior vertex-star cavity extraction failed: {failure:?}"),
            ))
        }
    };
    let boundary_node_ids = constrained_cavity_boundary_node_ids(&cavity);
    let mut star_refill = None;
    for apex in &boundary_node_ids {
        if let Some(tetrahedra) = constrained_cavity_star_refill_for_apex(&cavity, *apex, |index| {
            topology
                .nodes
                .get(index as usize)
                .map(|node| node.coordinates_m)
        }) {
            star_refill = Some(tetrahedra);
            break;
        }
    }
    let refill_tetrahedra = match star_refill.or(convex_delaunay_refill(
        topology,
        &cavity,
        &boundary_node_ids,
        insertion_options,
        cancellation,
    )?) {
        Some(tetrahedra) => tetrahedra,
        None => return Ok(None),
    };

    let source_identity = topology.nodes[source_index].identity;
    let nodes = topology
        .nodes
        .iter()
        .copied()
        .filter(|node| node.identity != source_identity)
        .collect::<Vec<_>>();
    let indices = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.identity, index as u32))
        .collect::<BTreeMap<_, _>>();
    let remap = |old: u32| -> Result<u32, DelaunayVolumeSliverError> {
        indices
            .get(&topology.nodes[old as usize].identity)
            .copied()
            .ok_or_else(|| {
                error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    "interior vertex-star cavity remap referenced the removed node",
                )
            })
    };
    let removed = selected.into_iter().collect::<BTreeSet<_>>();
    let mut tetrahedra = topology
        .tetrahedra
        .iter()
        .enumerate()
        .filter(|(index, _)| !removed.contains(index))
        .map(|(_, tetrahedron)| {
            Ok((
                tetrahedron
                    .vertex_indices
                    .map(remap)
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()?
                    .try_into()
                    .expect("tetrahedron remap retains four vertices"),
                tetrahedron.region_id.clone(),
            ))
        })
        .collect::<Result<Vec<_>, DelaunayVolumeSliverError>>()?;
    for tetrahedron in refill_tetrahedra {
        let remapped = tetrahedron
            .map(remap)
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .expect("cavity refill retains four vertices");
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
    let old_indices = topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.identity, index as u32))
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
            face.node_ids
                .map(|index| topology.nodes[index as usize].coordinates_m)
        })
        .collect::<Vec<_>>();
    let admitted = local
        .tetrahedra
        .iter()
        .filter(|tetrahedron| {
            let points = tetrahedron
                .vertex_indices
                .map(|index| local.nodes[index as usize].coordinates_m);
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
