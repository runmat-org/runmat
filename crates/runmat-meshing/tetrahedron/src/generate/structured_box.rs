use std::collections::{BTreeMap, BTreeSet};

mod boundary;
mod shape;

use super::evidence::record_input_plc_evidence;
use super::validation::validate_tetrahedron_generation_plc;
use super::{
    Tetrahedron4Element, TetrahedronBoundaryFace, TetrahedronGenerationError, TetrahedronMesh,
    TetrahedronMeshNode,
};
use boundary::exterior_boundary_faces;
use runmat_meshing_core::{
    contracts::{MeshingStage, ProtectedBoundaryComplex, StageEvidence, TopologyEntityId},
    quality::predicate::{tetrahedron_scaled_jacobian, tetrahedron_signed_volume},
};
use shape::{plc_nodes_are_box_corners, validate_structured_box_plc};

pub fn generate_structured_box_tetrahedron_mesh_from_plc(
    plc: &ProtectedBoundaryComplex,
) -> Result<TetrahedronMesh, TetrahedronGenerationError> {
    validate_tetrahedron_generation_plc(plc)?;

    let bounds = plc_bounds(plc)?;
    let tolerance = structured_box_tolerance(bounds);
    validate_structured_box_plc(plc, bounds, tolerance)?;
    if !plc.protected_edges.is_empty() || !plc_nodes_are_box_corners(plc, bounds, tolerance) {
        return generate_boundary_conforming_box_tetrahedron_mesh(plc, bounds, tolerance);
    }
    let material_region_ids = plc_material_region_ids(plc);

    let mut nodes = plc
        .nodes
        .iter()
        .map(|node| TetrahedronMeshNode {
            node_id: node.node_id.clone(),
            coordinates_m: node.coordinates_m,
        })
        .collect::<Vec<_>>();
    let corner_ids = structured_box_corner_nodes(bounds, &mut nodes, tolerance);
    let tetrahedron_corners = [
        [0, 1, 3, 7],
        [0, 3, 2, 7],
        [0, 2, 6, 7],
        [0, 6, 4, 7],
        [0, 4, 5, 7],
        [0, 5, 1, 7],
    ];
    let coordinates_by_id = nodes
        .iter()
        .map(|node| (node.node_id.clone(), node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut elements = Vec::<Tetrahedron4Element>::with_capacity(tetrahedron_corners.len());
    let mut min_scaled_jacobian = f64::INFINITY;
    for (tetrahedron_index, corners) in tetrahedron_corners.iter().enumerate() {
        let mut node_ids = corners.map(|corner| corner_ids[corner].clone());
        let points = node_ids.clone().map(|node_id| coordinates_by_id[&node_id]);
        if tetrahedron_signed_volume(points) < 0.0 {
            node_ids.swap(1, 2);
        }
        let points = node_ids.clone().map(|node_id| coordinates_by_id[&node_id]);
        min_scaled_jacobian = min_scaled_jacobian.min(tetrahedron_scaled_jacobian(points));
        elements.push(Tetrahedron4Element {
            element_id: TopologyEntityId {
                stage: MeshingStage::TetrahedronMesh,
                id: format!("structured_box_tetrahedron_{tetrahedron_index}"),
            },
            node_ids,
            material_region_id: material_region_ids[tetrahedron_index % material_region_ids.len()]
                .clone(),
        });
    }

    let boundary_faces =
        exterior_boundary_faces(&elements, &coordinates_by_id, plc, bounds, tolerance)?;

    let mut evidence = StageEvidence::complete(MeshingStage::TetrahedronMesh);
    evidence
        .entity_counts
        .insert("nodes".to_string(), nodes.len());
    evidence
        .entity_counts
        .insert("tetrahedron4_elements".to_string(), elements.len());
    evidence
        .entity_counts
        .insert("boundary_faces".to_string(), boundary_faces.len());
    evidence
        .entity_counts
        .insert("plc_boundary_nodes".to_string(), plc.nodes.len());
    record_input_plc_evidence(plc, &mut evidence);
    evidence.min_scaled_jacobian = Some(min_scaled_jacobian);

    Ok(TetrahedronMesh {
        mesh_id: "structured_box_tetrahedron_mesh".to_string(),
        nodes,
        elements,
        boundary_faces,
        recovery_complete: false,
        quality_optimized: false,
        evidence,
    })
}

fn generate_boundary_conforming_box_tetrahedron_mesh(
    plc: &ProtectedBoundaryComplex,
    bounds: [[f64; 3]; 2],
    tolerance: f64,
) -> Result<TetrahedronMesh, TetrahedronGenerationError> {
    let coordinates_by_id = plc
        .nodes
        .iter()
        .map(|node| (node.node_id.clone(), node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let interior_selection =
        select_boundary_conforming_box_interior(plc, &coordinates_by_id, bounds, tolerance)?;
    let interior = interior_selection.point;
    let interior_id = TopologyEntityId {
        stage: MeshingStage::TetrahedronMesh,
        id: "structured_box_interior_0".to_string(),
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

    let volume_epsilon = tolerance.powi(3).max(f64::EPSILON);
    let mut elements = Vec::<Tetrahedron4Element>::with_capacity(plc.facets.len());
    let mut min_scaled_jacobian = f64::INFINITY;
    for (element_index, facet) in plc.facets.iter().enumerate() {
        let mut node_ids = [
            facet.node_ids[0].clone(),
            facet.node_ids[1].clone(),
            facet.node_ids[2].clone(),
            interior_id.clone(),
        ];
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
            interior,
        ];
        let signed_volume = tetrahedron_signed_volume(points);
        if signed_volume.abs() <= volume_epsilon {
            return Err(TetrahedronGenerationError::DegenerateBoundaryFacet {
                facet_id: facet.facet_id.id.clone(),
            });
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
                id: format!("structured_box_boundary_conforming_tetrahedron_{element_index}"),
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
            source_edge_ids: super::source_edge_ids_for_face_edges(
                &plc.protected_edges,
                facet.node_ids.clone(),
            ),
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
    evidence.entity_counts.insert(
        "boundary_conforming_box_facets".to_string(),
        plc.facets.len(),
    );
    evidence.entity_counts.insert(
        "interior_smoothing_candidate_points".to_string(),
        interior_selection.candidate_count,
    );
    evidence.entity_counts.insert(
        "interior_smoothing_accepted_points".to_string(),
        usize::from(interior_selection.improved),
    );
    record_input_plc_evidence(plc, &mut evidence);
    evidence.min_scaled_jacobian = Some(min_scaled_jacobian);

    Ok(TetrahedronMesh {
        mesh_id: "structured_box_boundary_conforming_tetrahedron_mesh".to_string(),
        nodes,
        elements,
        boundary_faces,
        recovery_complete: false,
        quality_optimized: false,
        evidence,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct InteriorPointSelection {
    point: [f64; 3],
    min_scaled_jacobian: f64,
    candidate_count: usize,
    improved: bool,
}

fn select_boundary_conforming_box_interior(
    plc: &ProtectedBoundaryComplex,
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    bounds: [[f64; 3]; 2],
    tolerance: f64,
) -> Result<InteriorPointSelection, TetrahedronGenerationError> {
    let center = bounds_center(bounds);
    let volume_epsilon = tolerance.powi(3).max(f64::EPSILON);
    let baseline_quality = min_scaled_jacobian_for_boundary_conforming_interior(
        plc,
        coordinates_by_id,
        center,
        volume_epsilon,
    )?;
    let mut best = InteriorPointSelection {
        point: center,
        min_scaled_jacobian: baseline_quality,
        candidate_count: 0,
        improved: false,
    };
    let mut candidates = Vec::<[f64; 3]>::new();
    for x_fraction in [0.25, 0.35, 0.5, 0.65, 0.75] {
        for y_fraction in [0.25, 0.35, 0.5, 0.65, 0.75] {
            for z_fraction in [0.25, 0.35, 0.5, 0.65, 0.75] {
                candidates.push([
                    lerp(bounds[0][0], bounds[1][0], x_fraction),
                    lerp(bounds[0][1], bounds[1][1], y_fraction),
                    lerp(bounds[0][2], bounds[1][2], z_fraction),
                ]);
            }
        }
    }
    candidates.sort_by(|left, right| {
        left[0]
            .total_cmp(&right[0])
            .then_with(|| left[1].total_cmp(&right[1]))
            .then_with(|| left[2].total_cmp(&right[2]))
    });
    candidates
        .dedup_by(|left, right| (0..3).all(|axis| (left[axis] - right[axis]).abs() <= tolerance));
    best.candidate_count = candidates.len();

    for candidate in candidates {
        let Ok(min_scaled_jacobian) = min_scaled_jacobian_for_boundary_conforming_interior(
            plc,
            coordinates_by_id,
            candidate,
            volume_epsilon,
        ) else {
            continue;
        };
        if min_scaled_jacobian > best.min_scaled_jacobian {
            best.point = candidate;
            best.min_scaled_jacobian = min_scaled_jacobian;
            best.improved = min_scaled_jacobian > baseline_quality + 1.0e-12;
        }
    }

    Ok(best)
}

fn min_scaled_jacobian_for_boundary_conforming_interior(
    plc: &ProtectedBoundaryComplex,
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    interior: [f64; 3],
    volume_epsilon: f64,
) -> Result<f64, TetrahedronGenerationError> {
    let mut min_scaled_jacobian = f64::INFINITY;
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
            interior,
        ];
        if tetrahedron_signed_volume(points).abs() <= volume_epsilon {
            return Err(TetrahedronGenerationError::DegenerateBoundaryFacet {
                facet_id: facet.facet_id.id.clone(),
            });
        }
        min_scaled_jacobian = min_scaled_jacobian.min(tetrahedron_scaled_jacobian(points));
    }
    Ok(min_scaled_jacobian)
}

fn plc_bounds(plc: &ProtectedBoundaryComplex) -> Result<[[f64; 3]; 2], TetrahedronGenerationError> {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for node in &plc.nodes {
        if !node
            .coordinates_m
            .iter()
            .all(|coordinate| coordinate.is_finite())
        {
            return Err(TetrahedronGenerationError::NonFinitePlcNode {
                node_id: node.node_id.id.clone(),
            });
        }
        for axis in 0..3 {
            min[axis] = min[axis].min(node.coordinates_m[axis]);
            max[axis] = max[axis].max(node.coordinates_m[axis]);
        }
    }
    if (0..3).all(|axis| {
        min[axis].is_finite() && max[axis].is_finite() && max[axis] - min[axis] > f64::EPSILON
    }) {
        Ok([min, max])
    } else {
        Err(TetrahedronGenerationError::DegeneratePlcBounds)
    }
}

fn plc_material_region_ids(plc: &ProtectedBoundaryComplex) -> Vec<String> {
    let material_region_ids = plc
        .facets
        .iter()
        .flat_map(|facet| facet.material_interface_ids.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if material_region_ids.is_empty() {
        vec!["solid_body".to_string()]
    } else {
        material_region_ids
    }
}

fn structured_box_corner_nodes(
    bounds: [[f64; 3]; 2],
    nodes: &mut Vec<TetrahedronMeshNode>,
    tolerance: f64,
) -> [TopologyEntityId; 8] {
    let [min, max] = bounds;
    let corners = [
        [min[0], min[1], min[2]],
        [max[0], min[1], min[2]],
        [min[0], max[1], min[2]],
        [max[0], max[1], min[2]],
        [min[0], min[1], max[2]],
        [max[0], min[1], max[2]],
        [min[0], max[1], max[2]],
        [max[0], max[1], max[2]],
    ];
    corners.each_ref().map(|coordinates| {
        if let Some(node) = nodes
            .iter()
            .find(|node| same_point(node.coordinates_m, *coordinates, tolerance))
        {
            return node.node_id.clone();
        }
        let node_id = TopologyEntityId {
            stage: MeshingStage::TetrahedronMesh,
            id: format!("structured_box_node_{}", nodes.len()),
        };
        nodes.push(TetrahedronMeshNode {
            node_id: node_id.clone(),
            coordinates_m: *coordinates,
        });
        node_id
    })
}

fn structured_box_tolerance(bounds: [[f64; 3]; 2]) -> f64 {
    let [min, max] = bounds;
    ((max[0] - min[0])
        .abs()
        .max((max[1] - min[1]).abs())
        .max((max[2] - min[2]).abs())
        * 1.0e-9)
        .max(1.0e-12)
}

fn bounds_center(bounds: [[f64; 3]; 2]) -> [f64; 3] {
    [
        (bounds[0][0] + bounds[1][0]) * 0.5,
        (bounds[0][1] + bounds[1][1]) * 0.5,
        (bounds[0][2] + bounds[1][2]) * 0.5,
    ]
}

fn lerp(min: f64, max: f64, fraction: f64) -> f64 {
    min + (max - min) * fraction
}

fn same_point(left: [f64; 3], right: [f64; 3], tolerance: f64) -> bool {
    (0..3).all(|axis| (left[axis] - right[axis]).abs() <= tolerance)
}
