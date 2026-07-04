use std::collections::BTreeMap;

use runmat_meshing_core::{
    contracts::{ProtectedBoundaryComplex, TopologyEntityId},
    quality::predicate::{cross, dot, norm, sub},
    quality::tolerance::MeshingTolerance,
};

use super::super::TetrahedronGenerationError;

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

fn signed_side(distance: f64, tolerance: MeshingTolerance) -> i8 {
    if distance > tolerance.absolute_m {
        1
    } else if distance < -tolerance.absolute_m {
        -1
    } else {
        0
    }
}
