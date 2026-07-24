use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    quality::predicate::{
        point_in_closed_triangle_surface, tetrahedron_centroid, Point3, PointInClosedSurface,
    },
    quality::tolerance::MeshingTolerance,
};

use super::{
    boundary_nodes::{
        boundary_node_coordinates, candidate_respects_protected_boundary_distance,
        cavity_boundary_node_ids, cavity_boundary_triangles,
    },
    cap_connectors::append_cap_side_connector_chain_tetrahedra,
    exact_cover::exact_cover_refill_from_candidate_tetrahedra,
    refill_tetrahedra::raw_refill_tetrahedron_with_rejection_reason,
    topology::sorted_tetrahedron_nodes,
    validation::{validate_constrained_cavity, validate_refill_options},
    ConstrainedCavity, ConstrainedCavityNode, ConstrainedCavityRefill,
    ConstrainedCavityRefillError, ConstrainedCavityRefillOptions,
    ConstrainedCavityRefillTetrahedron, MAX_BOUNDARY_EXACT_COVER_CANDIDATES,
    MAX_BOUNDARY_EXACT_COVER_FACES, MAX_BOUNDARY_EXACT_COVER_NODES,
};

pub fn retriangulate_constrained_cavity_from_nodes(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<Option<ConstrainedCavityRefill>, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, nodes)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity);
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let tolerance = MeshingTolerance::default();
    let mut node_map = BTreeMap::<u32, Point3>::new();
    for node in nodes {
        if node_map.insert(node.node_id, node.coordinates_m).is_some() {
            return Err(ConstrainedCavityRefillError::DuplicateInteriorNode {
                node_id: node.node_id,
            });
        }
    }
    let mut candidate_nodes = Vec::<ConstrainedCavityNode>::new();
    for node in nodes {
        if boundary_node_ids.contains(&node.node_id) {
            candidate_nodes.push(node.clone());
            continue;
        }
        if !candidate_respects_protected_boundary_distance(
            cavity,
            &boundary_node_map,
            node.coordinates_m,
            options,
        ) {
            continue;
        }
        if point_in_closed_triangle_surface(node.coordinates_m, &boundary_triangles, tolerance)
            == PointInClosedSurface::Inside
        {
            candidate_nodes.push(node.clone());
        }
    }
    if candidate_nodes.len() < 4
        || candidate_nodes.len() > MAX_BOUNDARY_EXACT_COVER_NODES
        || cavity.boundary_faces.len() > MAX_BOUNDARY_EXACT_COVER_FACES
    {
        return Ok(None);
    }
    candidate_nodes.sort_by_key(|node| node.node_id);
    let mut candidate_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut seen_tetrahedra = BTreeSet::<[u32; 4]>::new();
    for first in 0..candidate_nodes.len() {
        for second in (first + 1)..candidate_nodes.len() {
            for third in (second + 1)..candidate_nodes.len() {
                for fourth in (third + 1)..candidate_nodes.len() {
                    let tetrahedron_node_ids = [
                        candidate_nodes[first].node_id,
                        candidate_nodes[second].node_id,
                        candidate_nodes[third].node_id,
                        candidate_nodes[fourth].node_id,
                    ];
                    if !seen_tetrahedra.insert(sorted_tetrahedron_nodes(tetrahedron_node_ids)) {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| node_map[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
                        tolerance,
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        options,
                    ) {
                        candidate_tetrahedra.push(tetrahedron);
                    }
                    if candidate_tetrahedra.len() > MAX_BOUNDARY_EXACT_COVER_CANDIDATES {
                        return Ok(None);
                    }
                }
            }
        }
    }
    let inserted_node_ids = candidate_nodes
        .iter()
        .filter_map(|node| (!boundary_node_ids.contains(&node.node_id)).then_some(node.node_id))
        .collect::<BTreeSet<_>>();
    if !inserted_node_ids.is_empty() {
        append_cap_side_connector_chain_tetrahedra(
            &mut candidate_tetrahedra,
            &mut seen_tetrahedra,
            &node_map,
            &inserted_node_ids,
            &boundary_triangles,
            options,
        );
        if candidate_tetrahedra.len() > MAX_BOUNDARY_EXACT_COVER_CANDIDATES {
            return Ok(None);
        }
    }
    let Some(mut refill) =
        exact_cover_refill_from_candidate_tetrahedra(cavity, &candidate_tetrahedra, options)
            .map_err(ConstrainedCavityRefillError::Validation)?
    else {
        return Ok(None);
    };
    let used_node_ids = refill
        .tetrahedra
        .iter()
        .flat_map(|tetrahedron| tetrahedron.node_ids)
        .collect::<BTreeSet<_>>();
    refill.inserted_nodes = candidate_nodes
        .into_iter()
        .filter(|node| !boundary_node_ids.contains(&node.node_id))
        .filter(|node| used_node_ids.contains(&node.node_id))
        .collect();
    Ok(Some(refill))
}
