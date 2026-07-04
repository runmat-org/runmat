use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{ProtectedBoundaryComplex, TopologyEntityId},
    quality::predicate::{cross, dot, norm, sub, tetrahedron_signed_volume},
    quality::tolerance::MeshingTolerance,
};

use super::super::TetrahedronGenerationError;
use super::bounds::bounds_span;

pub(super) fn validate_convex_boundary_facets(
    plc: &ProtectedBoundaryComplex,
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    interior: [f64; 3],
    tolerance: MeshingTolerance,
) -> Result<(), TetrahedronGenerationError> {
    for facet in &plc.facets {
        let points = [
            *coordinates_by_id.get(&facet.node_ids[0]).ok_or_else(|| {
                TetrahedronGenerationError::MissingPlcNode {
                    node_id: facet.node_ids[0].id.clone(),
                }
            })?,
            *coordinates_by_id.get(&facet.node_ids[1]).ok_or_else(|| {
                TetrahedronGenerationError::MissingPlcNode {
                    node_id: facet.node_ids[1].id.clone(),
                }
            })?,
            *coordinates_by_id.get(&facet.node_ids[2]).ok_or_else(|| {
                TetrahedronGenerationError::MissingPlcNode {
                    node_id: facet.node_ids[2].id.clone(),
                }
            })?,
        ];
        let normal = cross(sub(points[1], points[0]), sub(points[2], points[0]));
        let normal_norm = norm(normal);
        if normal_norm <= tolerance.absolute_m {
            return Err(TetrahedronGenerationError::DegenerateBoundaryFacet {
                facet_id: facet.facet_id.id.clone(),
            });
        }

        let mut boundary_side = 0_i8;
        for node in &plc.nodes {
            if facet.node_ids.contains(&node.node_id) {
                continue;
            }
            let signed_distance = dot(normal, sub(node.coordinates_m, points[0])) / normal_norm;
            let side = signed_side(signed_distance, tolerance);
            if side == 0 {
                continue;
            }
            if boundary_side == 0 {
                boundary_side = side;
            } else if boundary_side != side {
                return Err(TetrahedronGenerationError::UnsupportedConvexPolyhedronPlc);
            }
        }
        if boundary_side == 0 {
            return Err(TetrahedronGenerationError::DegenerateConvexPolyhedronPlc);
        }

        let interior_side = signed_side(
            dot(normal, sub(interior, points[0])) / normal_norm,
            tolerance,
        );
        if interior_side == 0 {
            return Err(TetrahedronGenerationError::DegenerateConvexPolyhedronPlc);
        }
        if interior_side != boundary_side {
            return Err(TetrahedronGenerationError::UnsupportedConvexPolyhedronPlc);
        }
    }
    Ok(())
}

pub(super) fn validate_boundary_nodes_are_hull_nodes(
    plc: &ProtectedBoundaryComplex,
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    bounds: [[f64; 3]; 2],
    tolerance: MeshingTolerance,
) -> Result<(), TetrahedronGenerationError> {
    if plc.nodes.len() < 5 {
        return Ok(());
    }
    let referenced_node_ids = plc
        .facets
        .iter()
        .flat_map(|facet| facet.node_ids.iter().cloned())
        .collect::<BTreeSet<_>>();
    if plc
        .nodes
        .iter()
        .any(|node| !referenced_node_ids.contains(&node.node_id))
    {
        return Err(TetrahedronGenerationError::UnsupportedConvexPolyhedronPlc);
    }

    let span = bounds_span(bounds).max(1.0);
    let volume_epsilon = tolerance.volume_epsilon(span);
    for node in &plc.nodes {
        let boundary_node = coordinates_by_id[&node.node_id];
        let other_nodes = plc
            .nodes
            .iter()
            .filter(|other| other.node_id != node.node_id)
            .map(|other| other.coordinates_m)
            .collect::<Vec<_>>();
        for first in 0..other_nodes.len() {
            for second in (first + 1)..other_nodes.len() {
                for third in (second + 1)..other_nodes.len() {
                    for fourth in (third + 1)..other_nodes.len() {
                        let tetrahedron = [
                            other_nodes[first],
                            other_nodes[second],
                            other_nodes[third],
                            other_nodes[fourth],
                        ];
                        if point_strictly_inside_tetrahedron(
                            boundary_node,
                            tetrahedron,
                            volume_epsilon,
                        ) {
                            return Err(TetrahedronGenerationError::UnsupportedConvexPolyhedronPlc);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn point_strictly_inside_tetrahedron(
    point: [f64; 3],
    tetrahedron: [[f64; 3]; 4],
    volume_epsilon: f64,
) -> bool {
    let total_volume = tetrahedron_signed_volume(tetrahedron).abs();
    if total_volume <= volume_epsilon {
        return false;
    }
    let sub_volumes = [
        tetrahedron_signed_volume([point, tetrahedron[1], tetrahedron[2], tetrahedron[3]]).abs(),
        tetrahedron_signed_volume([tetrahedron[0], point, tetrahedron[2], tetrahedron[3]]).abs(),
        tetrahedron_signed_volume([tetrahedron[0], tetrahedron[1], point, tetrahedron[3]]).abs(),
        tetrahedron_signed_volume([tetrahedron[0], tetrahedron[1], tetrahedron[2], point]).abs(),
    ];
    sub_volumes.iter().all(|volume| *volume > volume_epsilon)
        && (sub_volumes.iter().sum::<f64>() - total_volume).abs() <= volume_epsilon
}

fn signed_side(distance: f64, tolerance: MeshingTolerance) -> i8 {
    if distance > tolerance.absolute_m {
        1
    } else if distance < -tolerance.absolute_m {
        -1
    } else {
        0
    }
}
