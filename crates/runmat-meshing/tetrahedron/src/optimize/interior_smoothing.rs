use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{
        Tetrahedron4Element, TetrahedronMesh, TopologyEntityId,
        TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ACCEPTED_COUNT,
        TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ATTEMPT_COUNT,
        TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_BUDGET_LIMIT_COUNT,
        TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_REJECTED_COUNT,
        TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_REJECTION_PREFIX,
    },
    quality::predicate::{
        tetrahedron_edge_aspect_ratio, tetrahedron_scaled_jacobian, tetrahedron_volume, Point3,
    },
};
use runmat_meshing_opt::smooth::{
    evaluate_tetrahedron_smoothing_candidate, TetrahedronSmoothingOptions,
    TetrahedronSmoothingQuality, TetrahedronSmoothingRejectionReason,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TetrahedronMeshInteriorSmoothingOptions {
    pub smoothing: TetrahedronSmoothingOptions,
    pub max_attempted_points: usize,
    pub max_accepted_points: usize,
    pub relaxation: f64,
}

impl Default for TetrahedronMeshInteriorSmoothingOptions {
    fn default() -> Self {
        Self {
            smoothing: TetrahedronSmoothingOptions::default(),
            max_attempted_points: 32,
            max_accepted_points: 8,
            relaxation: 0.5,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TetrahedronMeshInteriorSmoothingReport {
    pub attempted_point_count: usize,
    pub accepted_point_count: usize,
    pub rejected_point_count: usize,
    pub budget_limited_point_count: usize,
    pub rejected_by_reason: BTreeMap<String, usize>,
}

pub fn smooth_tetrahedron_mesh_interior(
    mesh: &mut TetrahedronMesh,
    options: TetrahedronMeshInteriorSmoothingOptions,
) -> TetrahedronMeshInteriorSmoothingReport {
    let mut report = TetrahedronMeshInteriorSmoothingReport::default();
    if options.max_attempted_points == 0
        || options.max_accepted_points == 0
        || !options.relaxation.is_finite()
        || options.relaxation <= 0.0
        || mesh.elements.is_empty()
    {
        record_interior_smoothing_evidence(mesh, &report);
        return report;
    }

    let boundary_nodes = boundary_node_ids(mesh);
    let node_index = node_index(mesh);
    let incident_elements = incident_elements_by_node(mesh);
    let mut accepted = 0_usize;

    for node in mesh.nodes.clone() {
        if boundary_nodes.contains(&node.node_id) {
            continue;
        }
        if report.attempted_point_count >= options.max_attempted_points {
            report.budget_limited_point_count += 1;
            continue;
        }
        if accepted >= options.max_accepted_points {
            report.budget_limited_point_count += 1;
            continue;
        }
        let Some(element_indices) = incident_elements.get(&node.node_id) else {
            continue;
        };
        if element_indices.is_empty() {
            continue;
        }
        let current_quality = patch_quality(mesh, element_indices);
        let current_min_scaled_jacobian = current_quality
            .iter()
            .map(|quality| quality.scaled_jacobian)
            .fold(f64::INFINITY, f64::min);
        if current_min_scaled_jacobian >= options.smoothing.min_scaled_jacobian {
            continue;
        }
        report.attempted_point_count += 1;

        let Some(candidate_coordinates) =
            smoothed_node_coordinates(mesh, &node.node_id, element_indices, &node_index, options)
        else {
            record_interior_smoothing_rejection(&mut report, "missing_neighbor");
            continue;
        };
        let candidate_quality = patch_quality_with_relocated_node(
            mesh,
            element_indices,
            &node.node_id,
            candidate_coordinates,
        );
        match evaluate_tetrahedron_smoothing_candidate(
            &current_quality,
            &candidate_quality,
            options.smoothing,
        ) {
            Ok(evaluation) if evaluation.accepted => {
                if let Some(target) = mesh
                    .nodes
                    .iter_mut()
                    .find(|candidate| candidate.node_id == node.node_id)
                {
                    target.coordinates_m = candidate_coordinates;
                    report.accepted_point_count += 1;
                    accepted += 1;
                } else {
                    record_interior_smoothing_rejection(&mut report, "missing_node");
                }
            }
            Ok(evaluation) => {
                let reason = evaluation
                    .rejection_reason
                    .unwrap_or(TetrahedronSmoothingRejectionReason::QualityDoesNotImprove)
                    .as_str();
                record_interior_smoothing_rejection(&mut report, reason);
            }
            Err(_) => record_interior_smoothing_rejection(&mut report, "invalid_options"),
        }
    }

    if report.attempted_point_count > 0 {
        mesh.quality_optimized = true;
    }
    record_interior_smoothing_evidence(mesh, &report);
    report
}

fn boundary_node_ids(mesh: &TetrahedronMesh) -> BTreeSet<TopologyEntityId> {
    mesh.boundary_faces
        .iter()
        .flat_map(|face| face.node_ids.iter().cloned())
        .collect()
}

fn node_index(mesh: &TetrahedronMesh) -> BTreeMap<TopologyEntityId, Point3> {
    mesh.nodes
        .iter()
        .map(|node| (node.node_id.clone(), node.coordinates_m))
        .collect()
}

fn incident_elements_by_node(mesh: &TetrahedronMesh) -> BTreeMap<TopologyEntityId, Vec<usize>> {
    let mut incident_elements = BTreeMap::<TopologyEntityId, Vec<usize>>::new();
    for (index, element) in mesh.elements.iter().enumerate() {
        for node_id in &element.node_ids {
            incident_elements
                .entry(node_id.clone())
                .or_default()
                .push(index);
        }
    }
    incident_elements
}

fn smoothed_node_coordinates(
    mesh: &TetrahedronMesh,
    node_id: &TopologyEntityId,
    element_indices: &[usize],
    node_coordinates: &BTreeMap<TopologyEntityId, Point3>,
    options: TetrahedronMeshInteriorSmoothingOptions,
) -> Option<Point3> {
    let current = node_coordinates.get(node_id)?;
    let mut neighbors = BTreeSet::<TopologyEntityId>::new();
    for element_index in element_indices {
        for neighbor_id in &mesh.elements[*element_index].node_ids {
            if neighbor_id != node_id {
                neighbors.insert(neighbor_id.clone());
            }
        }
    }
    if neighbors.is_empty() {
        return None;
    }
    let mut average = [0.0, 0.0, 0.0];
    let mut count = 0_usize;
    for neighbor_id in neighbors {
        let neighbor = node_coordinates.get(&neighbor_id)?;
        average[0] += neighbor[0];
        average[1] += neighbor[1];
        average[2] += neighbor[2];
        count += 1;
    }
    let count = count as f64;
    average[0] /= count;
    average[1] /= count;
    average[2] /= count;
    Some([
        current[0] + options.relaxation * (average[0] - current[0]),
        current[1] + options.relaxation * (average[1] - current[1]),
        current[2] + options.relaxation * (average[2] - current[2]),
    ])
}

fn patch_quality(
    mesh: &TetrahedronMesh,
    element_indices: &[usize],
) -> Vec<TetrahedronSmoothingQuality> {
    element_indices
        .iter()
        .map(|element_index| {
            tetrahedron_quality(*element_index, &mesh.elements[*element_index], mesh)
        })
        .collect()
}

fn patch_quality_with_relocated_node(
    mesh: &TetrahedronMesh,
    element_indices: &[usize],
    relocated_node_id: &TopologyEntityId,
    relocated_coordinates: Point3,
) -> Vec<TetrahedronSmoothingQuality> {
    element_indices
        .iter()
        .map(|element_index| {
            tetrahedron_quality_with_relocated_node(
                *element_index,
                &mesh.elements[*element_index],
                mesh,
                relocated_node_id,
                relocated_coordinates,
            )
        })
        .collect()
}

fn tetrahedron_quality(
    element_index: usize,
    element: &Tetrahedron4Element,
    mesh: &TetrahedronMesh,
) -> TetrahedronSmoothingQuality {
    let points = element.node_ids.clone().map(|node_id| {
        mesh.nodes
            .iter()
            .find(|node| node.node_id == node_id)
            .map(|node| node.coordinates_m)
            .unwrap_or([f64::NAN, f64::NAN, f64::NAN])
    });
    smoothing_quality(element_index, points)
}

fn tetrahedron_quality_with_relocated_node(
    element_index: usize,
    element: &Tetrahedron4Element,
    mesh: &TetrahedronMesh,
    relocated_node_id: &TopologyEntityId,
    relocated_coordinates: Point3,
) -> TetrahedronSmoothingQuality {
    let points = element.node_ids.clone().map(|node_id| {
        if &node_id == relocated_node_id {
            relocated_coordinates
        } else {
            mesh.nodes
                .iter()
                .find(|node| node.node_id == node_id)
                .map(|node| node.coordinates_m)
                .unwrap_or([f64::NAN, f64::NAN, f64::NAN])
        }
    });
    smoothing_quality(element_index, points)
}

fn smoothing_quality(element_index: usize, points: [Point3; 4]) -> TetrahedronSmoothingQuality {
    TetrahedronSmoothingQuality {
        tetrahedron_id: element_index as u32,
        volume_m3: tetrahedron_volume(points),
        scaled_jacobian: tetrahedron_scaled_jacobian(points),
        aspect_ratio: tetrahedron_edge_aspect_ratio(points),
    }
}

fn record_interior_smoothing_rejection(
    report: &mut TetrahedronMeshInteriorSmoothingReport,
    reason: &str,
) {
    report.rejected_point_count += 1;
    *report
        .rejected_by_reason
        .entry(reason.to_string())
        .or_default() += 1;
}

fn record_interior_smoothing_evidence(
    mesh: &mut TetrahedronMesh,
    report: &TetrahedronMeshInteriorSmoothingReport,
) {
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ATTEMPT_COUNT.to_string())
        .or_default() += report.attempted_point_count;
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ACCEPTED_COUNT.to_string())
        .or_default() += report.accepted_point_count;
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_REJECTED_COUNT.to_string())
        .or_default() += report.rejected_point_count;
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_BUDGET_LIMIT_COUNT.to_string())
        .or_default() += report.budget_limited_point_count;
    for (reason, count) in &report.rejected_by_reason {
        *mesh
            .evidence
            .rejection_counts
            .entry(format!(
                "{TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_REJECTION_PREFIX}{reason}"
            ))
            .or_default() += count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_meshing_core::contracts::{
        MeshingStage, StageEvidence, TetrahedronBoundaryFace, TetrahedronMeshNode,
    };

    #[test]
    fn interior_smoothing_accepts_quality_improving_node_move() {
        let mut mesh = smoothing_fixture([0.2, 0.2, 0.02]);

        let report = smooth_tetrahedron_mesh_interior(
            &mut mesh,
            TetrahedronMeshInteriorSmoothingOptions {
                smoothing: TetrahedronSmoothingOptions {
                    min_volume_m3: 1.0e-18,
                    min_scaled_jacobian: 0.15,
                    min_scaled_jacobian_improvement: 1.0e-12,
                    max_aspect_ratio_growth: 10.0,
                },
                max_attempted_points: 4,
                max_accepted_points: 1,
                relaxation: 0.5,
            },
        );

        assert_eq!(report.attempted_point_count, 1);
        assert_eq!(report.accepted_point_count, 1);
        assert_eq!(report.rejected_point_count, 0);
        assert!(mesh.quality_optimized);
        assert_ne!(mesh.nodes[4].coordinates_m, [0.2, 0.2, 0.02]);
        assert_eq!(
            mesh.evidence.entity_counts[TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ATTEMPT_COUNT],
            1
        );
        assert_eq!(
            mesh.evidence.entity_counts[TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ACCEPTED_COUNT],
            1
        );
    }

    #[test]
    fn interior_smoothing_records_rejected_candidate() {
        let mut mesh = smoothing_fixture([0.25, 0.25, 0.25]);

        let report = smooth_tetrahedron_mesh_interior(
            &mut mesh,
            TetrahedronMeshInteriorSmoothingOptions {
                smoothing: TetrahedronSmoothingOptions {
                    min_scaled_jacobian: 0.95,
                    max_aspect_ratio_growth: 10.0,
                    ..TetrahedronSmoothingOptions::default()
                },
                max_attempted_points: 4,
                max_accepted_points: 1,
                relaxation: 0.5,
            },
        );

        assert_eq!(report.attempted_point_count, 1);
        assert_eq!(report.accepted_point_count, 0);
        assert_eq!(report.rejected_point_count, 1);
        assert!(mesh.quality_optimized);
        assert_eq!(
            mesh.evidence.entity_counts[TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_REJECTED_COUNT],
            1
        );
        assert_eq!(
            mesh.evidence.rejection_counts[&format!(
                "{TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_REJECTION_PREFIX}scaled_jacobian_below_threshold"
            )],
            1
        );
    }

    #[test]
    fn interior_smoothing_skips_boundary_nodes() {
        let mut mesh = smoothing_fixture([0.2, 0.2, 0.02]);
        mesh.boundary_faces.push(TetrahedronBoundaryFace {
            face_id: entity("boundary"),
            node_ids: [entity("4"), entity("0"), entity("1")],
            source_face_id: entity("face"),
            source_edge_ids: [None, None, None],
        });

        let report = smooth_tetrahedron_mesh_interior(
            &mut mesh,
            TetrahedronMeshInteriorSmoothingOptions {
                smoothing: TetrahedronSmoothingOptions {
                    min_scaled_jacobian: 0.95,
                    max_aspect_ratio_growth: 10.0,
                    ..TetrahedronSmoothingOptions::default()
                },
                max_attempted_points: 4,
                max_accepted_points: 1,
                relaxation: 0.5,
            },
        );

        assert_eq!(report.attempted_point_count, 0);
        assert!(!mesh.quality_optimized);
    }

    fn smoothing_fixture(interior: Point3) -> TetrahedronMesh {
        TetrahedronMesh {
            mesh_id: "interior_smoothing_fixture".to_string(),
            tetrahedron_generation_family: "unknown".to_string(),
            nodes: vec![
                node("0", [0.0, 0.0, 0.0]),
                node("1", [1.0, 0.0, 0.0]),
                node("2", [0.0, 1.0, 0.0]),
                node("3", [0.0, 0.0, 1.0]),
                node("4", interior),
            ],
            elements: vec![
                element("0", ["4", "0", "1", "2"]),
                element("1", ["4", "0", "1", "3"]),
                element("2", ["4", "0", "2", "3"]),
                element("3", ["4", "1", "2", "3"]),
            ],
            boundary_faces: vec![
                boundary_face("boundary_0", ["0", "1", "2"]),
                boundary_face("boundary_1", ["0", "1", "3"]),
                boundary_face("boundary_2", ["0", "2", "3"]),
                boundary_face("boundary_3", ["1", "2", "3"]),
            ],
            recovery_complete: true,
            quality_optimized: false,
            evidence: StageEvidence::complete(MeshingStage::TetrahedronMesh),
        }
    }

    fn element(id: &str, node_ids: [&str; 4]) -> Tetrahedron4Element {
        Tetrahedron4Element {
            element_id: entity(id),
            node_ids: node_ids.map(entity),
            material_region_id: "body".to_string(),
        }
    }

    fn boundary_face(id: &str, node_ids: [&str; 3]) -> TetrahedronBoundaryFace {
        TetrahedronBoundaryFace {
            face_id: entity(id),
            node_ids: node_ids.map(entity),
            source_face_id: entity(id),
            source_edge_ids: [None, None, None],
        }
    }

    fn node(id: &str, coordinates_m: Point3) -> TetrahedronMeshNode {
        TetrahedronMeshNode {
            node_id: entity(id),
            coordinates_m,
        }
    }

    fn entity(id: &str) -> TopologyEntityId {
        TopologyEntityId {
            stage: MeshingStage::TetrahedronMesh,
            id: id.to_string(),
        }
    }
}
