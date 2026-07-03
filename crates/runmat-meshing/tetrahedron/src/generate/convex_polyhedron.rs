use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{MeshingStage, ProtectedBoundaryComplex, StageEvidence, TopologyEntityId},
    quality::predicate::{
        cross, dot, norm, sub, tetrahedron_scaled_jacobian, tetrahedron_signed_volume,
    },
    quality::tolerance::MeshingTolerance,
};

use super::validation::validate_tetrahedron_generation_plc;
use super::{
    Tetrahedron4Element, TetrahedronBoundaryFace, TetrahedronGenerationError, TetrahedronMesh,
    TetrahedronMeshNode,
};

pub fn generate_convex_polyhedron_tetrahedron_mesh_from_plc(
    plc: &ProtectedBoundaryComplex,
) -> Result<TetrahedronMesh, TetrahedronGenerationError> {
    validate_tetrahedron_generation_plc(plc)?;
    if plc.nodes.len() < 4 || plc.facets.len() < 4 {
        return Err(TetrahedronGenerationError::UnsupportedConvexPolyhedronPlc);
    }

    let (coordinates_by_id, bounds) = plc_coordinates_and_bounds(plc)?;
    let tolerance = MeshingTolerance::from_bounds(bounds[0], bounds[1]);
    let interior = plc_node_average(plc)?;
    validate_convex_boundary_facets(plc, &coordinates_by_id, interior, tolerance)?;
    validate_boundary_nodes_are_hull_nodes(plc, &coordinates_by_id, bounds, tolerance)?;

    let interior_id = TopologyEntityId {
        stage: MeshingStage::TetrahedronMesh,
        id: "convex_polyhedron_interior_0".to_string(),
    };
    let mut nodes = plc
        .nodes
        .iter()
        .map(|node| TetrahedronMeshNode {
            node_id: node.node_id.clone(),
            coordinates_m: node.coordinates_m,
        })
        .collect::<Vec<_>>();
    nodes.push(TetrahedronMeshNode {
        node_id: interior_id.clone(),
        coordinates_m: interior,
    });

    let mut elements = Vec::<Tetrahedron4Element>::with_capacity(plc.facets.len());
    let mut min_scaled_jacobian = f64::INFINITY;
    let span = bounds_span(bounds).max(1.0);
    let volume_epsilon = tolerance.volume_epsilon(span);
    for (element_index, facet) in plc.facets.iter().enumerate() {
        let mut node_ids = [
            facet.node_ids[0].clone(),
            facet.node_ids[1].clone(),
            facet.node_ids[2].clone(),
            interior_id.clone(),
        ];
        let points = [
            coordinates_by_id[&facet.node_ids[0]],
            coordinates_by_id[&facet.node_ids[1]],
            coordinates_by_id[&facet.node_ids[2]],
            interior,
        ];
        let signed_volume = tetrahedron_signed_volume(points);
        if signed_volume.abs() <= volume_epsilon {
            return Err(TetrahedronGenerationError::DegenerateConvexPolyhedronPlc);
        }
        if signed_volume < 0.0 {
            node_ids.swap(1, 2);
        }
        let points = node_ids.clone().map(|node_id| {
            if node_id == interior_id {
                interior
            } else {
                coordinates_by_id[&node_id]
            }
        });
        min_scaled_jacobian = min_scaled_jacobian.min(tetrahedron_scaled_jacobian(points));
        elements.push(Tetrahedron4Element {
            element_id: TopologyEntityId {
                stage: MeshingStage::TetrahedronMesh,
                id: format!("convex_polyhedron_tetrahedron_{element_index}"),
            },
            node_ids,
            material_region_id: facet
                .material_interface_ids
                .first()
                .cloned()
                .unwrap_or_else(|| "solid_body".to_string()),
        });
    }

    let boundary_faces = plc
        .facets
        .iter()
        .map(|facet| TetrahedronBoundaryFace {
            face_id: facet.facet_id.clone(),
            node_ids: facet.node_ids.clone(),
            source_face_id: facet.source_face_id.clone(),
        })
        .collect::<Vec<_>>();

    let mut evidence = StageEvidence::complete(MeshingStage::TetrahedronMesh);
    evidence
        .entity_counts
        .insert("nodes".to_string(), nodes.len());
    evidence
        .entity_counts
        .insert("interior_nodes".to_string(), 1);
    evidence
        .entity_counts
        .insert("tetrahedron4_elements".to_string(), elements.len());
    evidence
        .entity_counts
        .insert("boundary_faces".to_string(), boundary_faces.len());
    evidence
        .entity_counts
        .insert("plc_boundary_nodes".to_string(), plc.nodes.len());
    evidence.min_scaled_jacobian = Some(min_scaled_jacobian);

    Ok(TetrahedronMesh {
        mesh_id: "convex_polyhedron_tetrahedron_mesh".to_string(),
        nodes,
        elements,
        boundary_faces,
        recovery_complete: false,
        quality_optimized: false,
        evidence,
    })
}

fn plc_coordinates_and_bounds(
    plc: &ProtectedBoundaryComplex,
) -> Result<(BTreeMap<TopologyEntityId, [f64; 3]>, [[f64; 3]; 2]), TetrahedronGenerationError> {
    let mut coordinates_by_id = BTreeMap::<TopologyEntityId, [f64; 3]>::new();
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for node in &plc.nodes {
        if node
            .coordinates_m
            .iter()
            .any(|coordinate| !coordinate.is_finite())
        {
            return Err(TetrahedronGenerationError::NonFinitePlcNode {
                node_id: node.node_id.id.clone(),
            });
        }
        for axis in 0..3 {
            min[axis] = min[axis].min(node.coordinates_m[axis]);
            max[axis] = max[axis].max(node.coordinates_m[axis]);
        }
        coordinates_by_id.insert(node.node_id.clone(), node.coordinates_m);
    }
    if bounds_span([min, max]) <= f64::EPSILON {
        return Err(TetrahedronGenerationError::DegeneratePlcBounds);
    }
    Ok((coordinates_by_id, [min, max]))
}

fn validate_convex_boundary_facets(
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

fn validate_boundary_nodes_are_hull_nodes(
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

fn plc_node_average(
    plc: &ProtectedBoundaryComplex,
) -> Result<[f64; 3], TetrahedronGenerationError> {
    let mut sum = [0.0; 3];
    for node in &plc.nodes {
        for (axis, coordinate) in node.coordinates_m.iter().enumerate() {
            sum[axis] += coordinate;
        }
    }
    let count = plc.nodes.len() as f64;
    let interior = [sum[0] / count, sum[1] / count, sum[2] / count];
    if interior.iter().all(|coordinate| coordinate.is_finite()) {
        Ok(interior)
    } else {
        Err(TetrahedronGenerationError::NonFiniteInteriorPoint)
    }
}

fn bounds_span(bounds: [[f64; 3]; 2]) -> f64 {
    (0..3)
        .map(|axis| bounds[1][axis] - bounds[0][axis])
        .filter(|span| span.is_finite())
        .fold(0.0_f64, f64::max)
}
