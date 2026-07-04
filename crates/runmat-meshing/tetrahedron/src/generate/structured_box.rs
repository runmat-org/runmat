use std::collections::{BTreeMap, BTreeSet};

use super::validation::validate_tetrahedron_generation_plc;
use super::{
    Tetrahedron4Element, TetrahedronBoundaryFace, TetrahedronGenerationError, TetrahedronMesh,
    TetrahedronMeshNode,
};
use runmat_meshing_core::{
    contracts::{MeshingStage, ProtectedBoundaryComplex, StageEvidence, TopologyEntityId},
    quality::predicate::{tetrahedron_scaled_jacobian, tetrahedron_signed_volume},
};

pub fn generate_structured_box_tetrahedron_mesh_from_plc(
    plc: &ProtectedBoundaryComplex,
) -> Result<TetrahedronMesh, TetrahedronGenerationError> {
    validate_tetrahedron_generation_plc(plc)?;

    let bounds = plc_bounds(plc)?;
    let tolerance = structured_box_tolerance(bounds);
    validate_structured_box_plc(plc, bounds, tolerance)?;
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

fn validate_structured_box_plc(
    plc: &ProtectedBoundaryComplex,
    bounds: [[f64; 3]; 2],
    tolerance: f64,
) -> Result<(), TetrahedronGenerationError> {
    let [min, max] = bounds;
    let coordinates_by_id = plc
        .nodes
        .iter()
        .map(|node| (node.node_id.clone(), node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut covered_sides = [false; 6];
    for facet in &plc.facets {
        let coordinates = facet
            .node_ids
            .iter()
            .map(|node_id| {
                coordinates_by_id.get(node_id).copied().ok_or_else(|| {
                    TetrahedronGenerationError::MissingPlcNode {
                        node_id: node_id.id.clone(),
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let side_index = structured_box_side_index(&coordinates, min, max, tolerance)
            .ok_or(TetrahedronGenerationError::UnsupportedStructuredBoxPlc)?;
        covered_sides[side_index] = true;
    }
    if covered_sides.iter().all(|covered| *covered) {
        Ok(())
    } else {
        Err(TetrahedronGenerationError::UnsupportedStructuredBoxPlc)
    }
}

fn structured_box_side_index(
    coordinates: &[[f64; 3]],
    min: [f64; 3],
    max: [f64; 3],
    tolerance: f64,
) -> Option<usize> {
    for axis in 0..3 {
        if coordinates
            .iter()
            .all(|point| (point[axis] - min[axis]).abs() <= tolerance)
        {
            return Some(axis * 2);
        }
        if coordinates
            .iter()
            .all(|point| (point[axis] - max[axis]).abs() <= tolerance)
        {
            return Some(axis * 2 + 1);
        }
    }
    None
}

fn exterior_boundary_faces(
    elements: &[Tetrahedron4Element],
    coordinates_by_id: &BTreeMap<TopologyEntityId, [f64; 3]>,
    plc: &ProtectedBoundaryComplex,
    bounds: [[f64; 3]; 2],
    tolerance: f64,
) -> Result<Vec<TetrahedronBoundaryFace>, TetrahedronGenerationError> {
    let [min, max] = bounds;
    let mut plc_source_by_facet = BTreeMap::<[TopologyEntityId; 3], TopologyEntityId>::new();
    let mut plc_source_by_side = BTreeMap::<usize, TopologyEntityId>::new();
    for facet in &plc.facets {
        let coordinates = facet
            .node_ids
            .iter()
            .map(|node_id| {
                coordinates_by_id.get(node_id).copied().ok_or_else(|| {
                    TetrahedronGenerationError::MissingPlcNode {
                        node_id: node_id.id.clone(),
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let side_index = structured_box_side_index(&coordinates, min, max, tolerance)
            .ok_or(TetrahedronGenerationError::UnsupportedStructuredBoxPlc)?;
        plc_source_by_side
            .entry(side_index)
            .or_insert_with(|| facet.source_face_id.clone());
        plc_source_by_facet.insert(
            sorted_face(facet.node_ids.clone()),
            facet.source_face_id.clone(),
        );
    }

    let mut face_counts = BTreeMap::<[TopologyEntityId; 3], ([TopologyEntityId; 3], usize)>::new();
    for element in elements {
        for face in tetrahedron_faces(element.node_ids.clone()) {
            face_counts
                .entry(sorted_face(face.clone()))
                .and_modify(|(_, count)| *count += 1)
                .or_insert((face, 1));
        }
    }

    let mut boundary_faces = Vec::<TetrahedronBoundaryFace>::new();
    for (boundary_face_index, (face_key, (node_ids, count))) in face_counts.into_iter().enumerate()
    {
        if count != 1 {
            continue;
        }
        let source_face_id = match plc_source_by_facet.get(&face_key) {
            Some(source_face_id) => source_face_id.clone(),
            None => {
                let coordinates = node_ids
                    .iter()
                    .map(|node_id| coordinates_by_id[node_id])
                    .collect::<Vec<_>>();
                let side_index = structured_box_side_index(&coordinates, min, max, tolerance)
                    .ok_or(TetrahedronGenerationError::UnsupportedStructuredBoxPlc)?;
                plc_source_by_side
                    .get(&side_index)
                    .cloned()
                    .ok_or(TetrahedronGenerationError::UnsupportedStructuredBoxPlc)?
            }
        };
        boundary_faces.push(TetrahedronBoundaryFace {
            face_id: TopologyEntityId {
                stage: MeshingStage::TetrahedronMesh,
                id: format!("structured_box_boundary_face_{boundary_face_index}"),
            },
            node_ids,
            source_face_id,
        });
    }

    Ok(boundary_faces)
}

fn tetrahedron_faces(node_ids: [TopologyEntityId; 4]) -> [[TopologyEntityId; 3]; 4] {
    [
        [
            node_ids[0].clone(),
            node_ids[1].clone(),
            node_ids[2].clone(),
        ],
        [
            node_ids[0].clone(),
            node_ids[1].clone(),
            node_ids[3].clone(),
        ],
        [
            node_ids[0].clone(),
            node_ids[2].clone(),
            node_ids[3].clone(),
        ],
        [
            node_ids[1].clone(),
            node_ids[2].clone(),
            node_ids[3].clone(),
        ],
    ]
}

fn sorted_face(mut node_ids: [TopologyEntityId; 3]) -> [TopologyEntityId; 3] {
    node_ids.sort();
    node_ids
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

fn same_point(left: [f64; 3], right: [f64; 3], tolerance: f64) -> bool {
    (0..3).all(|axis| (left[axis] - right[axis]).abs() <= tolerance)
}
