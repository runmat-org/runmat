use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{
        Tetrahedron4Element, TetrahedronBoundaryFace, TetrahedronMesh, TopologyEntityId,
        TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ACCEPTED_COUNT,
        TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ATTEMPT_COUNT,
        TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_BUDGET_LIMIT_COUNT,
        TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_REJECTED_COUNT,
        TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_REJECTION_PREFIX,
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
pub struct TetrahedronMeshBoundarySmoothingOptions {
    pub smoothing: TetrahedronSmoothingOptions,
    pub max_attempted_points: usize,
    pub max_accepted_points: usize,
    pub relaxation: f64,
    pub max_projection_distance_m: f64,
}

impl Default for TetrahedronMeshBoundarySmoothingOptions {
    fn default() -> Self {
        Self {
            smoothing: TetrahedronSmoothingOptions::default(),
            max_attempted_points: 32,
            max_accepted_points: 8,
            relaxation: 0.35,
            max_projection_distance_m: 1.0,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TetrahedronMeshBoundarySmoothingReport {
    pub attempted_point_count: usize,
    pub accepted_point_count: usize,
    pub rejected_point_count: usize,
    pub budget_limited_point_count: usize,
    pub rejected_by_reason: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TetrahedronBoundarySmoothingProjection {
    pub point_m: Point3,
    pub distance_m: f64,
    pub in_bounds: bool,
}

pub trait TetrahedronBoundarySmoothingProjector {
    fn project_to_source_face(
        &self,
        source_face_id: &TopologyEntityId,
        point_m: Point3,
    ) -> Option<TetrahedronBoundarySmoothingProjection>;
}

pub fn smooth_tetrahedron_mesh_boundary_with_projector(
    mesh: &mut TetrahedronMesh,
    projector: &impl TetrahedronBoundarySmoothingProjector,
    options: TetrahedronMeshBoundarySmoothingOptions,
) -> TetrahedronMeshBoundarySmoothingReport {
    let mut report = TetrahedronMeshBoundarySmoothingReport::default();
    if options.max_attempted_points == 0
        || options.max_accepted_points == 0
        || !options.relaxation.is_finite()
        || options.relaxation <= 0.0
        || !options.max_projection_distance_m.is_finite()
        || options.max_projection_distance_m < 0.0
        || mesh.elements.is_empty()
    {
        record_boundary_smoothing_evidence(mesh, &report);
        return report;
    }

    let boundary_ownership = boundary_node_ownership(mesh);
    let node_index = node_index(mesh);
    let incident_elements = incident_elements_by_node(mesh);
    let mut accepted = 0_usize;

    for node in mesh.nodes.clone() {
        let Some(ownership) = boundary_ownership.get(&node.node_id) else {
            continue;
        };
        if ownership.protected_edge_node || ownership.source_face_ids.len() != 1 {
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

        let Some(raw_candidate) =
            smoothed_node_coordinates(mesh, &node.node_id, element_indices, &node_index, options)
        else {
            record_boundary_smoothing_rejection(&mut report, "missing_neighbor");
            continue;
        };
        let source_face_id = ownership
            .source_face_ids
            .iter()
            .next()
            .expect("source face count was checked");
        let Some(projection) = projector.project_to_source_face(source_face_id, raw_candidate)
        else {
            record_boundary_smoothing_rejection(&mut report, "missing_projection");
            continue;
        };
        if !projection.in_bounds {
            record_boundary_smoothing_rejection(&mut report, "projection_out_of_bounds");
            continue;
        }
        if !projection.distance_m.is_finite()
            || projection.distance_m > options.max_projection_distance_m
        {
            record_boundary_smoothing_rejection(&mut report, "projection_distance");
            continue;
        }

        let candidate_quality = patch_quality_with_relocated_node(
            mesh,
            element_indices,
            &node.node_id,
            projection.point_m,
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
                    target.coordinates_m = projection.point_m;
                    report.accepted_point_count += 1;
                    accepted += 1;
                } else {
                    record_boundary_smoothing_rejection(&mut report, "missing_node");
                }
            }
            Ok(evaluation) => {
                let reason = evaluation
                    .rejection_reason
                    .unwrap_or(TetrahedronSmoothingRejectionReason::QualityDoesNotImprove)
                    .as_str();
                record_boundary_smoothing_rejection(&mut report, reason);
            }
            Err(_) => record_boundary_smoothing_rejection(&mut report, "invalid_options"),
        }
    }

    if report.attempted_point_count > 0 {
        mesh.quality_optimized = true;
    }
    record_boundary_smoothing_evidence(mesh, &report);
    report
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct BoundaryNodeOwnership {
    source_face_ids: BTreeSet<TopologyEntityId>,
    protected_edge_node: bool,
}

fn boundary_node_ownership(
    mesh: &TetrahedronMesh,
) -> BTreeMap<TopologyEntityId, BoundaryNodeOwnership> {
    let mut ownership = BTreeMap::<TopologyEntityId, BoundaryNodeOwnership>::new();
    for face in &mesh.boundary_faces {
        let face_edges = boundary_face_edges(face);
        for node_id in &face.node_ids {
            let entry = ownership.entry(node_id.clone()).or_default();
            entry.source_face_ids.insert(face.source_face_id.clone());
            for (edge_index, edge) in face_edges.iter().enumerate() {
                if edge.contains(node_id) && face.source_edge_ids[edge_index].is_some() {
                    entry.protected_edge_node = true;
                }
            }
        }
    }
    ownership
}

fn boundary_face_edges(face: &TetrahedronBoundaryFace) -> [[TopologyEntityId; 2]; 3] {
    [
        sorted_edge([face.node_ids[0].clone(), face.node_ids[1].clone()]),
        sorted_edge([face.node_ids[1].clone(), face.node_ids[2].clone()]),
        sorted_edge([face.node_ids[2].clone(), face.node_ids[0].clone()]),
    ]
}

fn sorted_edge(mut edge: [TopologyEntityId; 2]) -> [TopologyEntityId; 2] {
    edge.sort();
    edge
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
    options: TetrahedronMeshBoundarySmoothingOptions,
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

fn record_boundary_smoothing_rejection(
    report: &mut TetrahedronMeshBoundarySmoothingReport,
    reason: &str,
) {
    report.rejected_point_count += 1;
    *report
        .rejected_by_reason
        .entry(reason.to_string())
        .or_default() += 1;
}

fn record_boundary_smoothing_evidence(
    mesh: &mut TetrahedronMesh,
    report: &TetrahedronMeshBoundarySmoothingReport,
) {
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ATTEMPT_COUNT.to_string())
        .or_default() += report.attempted_point_count;
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ACCEPTED_COUNT.to_string())
        .or_default() += report.accepted_point_count;
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_REJECTED_COUNT.to_string())
        .or_default() += report.rejected_point_count;
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_BUDGET_LIMIT_COUNT.to_string())
        .or_default() += report.budget_limited_point_count;
    for (reason, count) in &report.rejected_by_reason {
        *mesh
            .evidence
            .rejection_counts
            .entry(format!(
                "{TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_REJECTION_PREFIX}{reason}"
            ))
            .or_default() += count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_meshing_core::contracts::{MeshingStage, StageEvidence, TetrahedronMeshNode};

    #[test]
    fn boundary_smoothing_accepts_quality_improving_face_projected_move() {
        let mut mesh =
            boundary_smoothing_fixture([0.02, 0.02, 0.0], [Some(entity("edge_0")), None, None]);
        let projector = PlanarProjector;

        let report = smooth_tetrahedron_mesh_boundary_with_projector(
            &mut mesh,
            &projector,
            TetrahedronMeshBoundarySmoothingOptions {
                smoothing: TetrahedronSmoothingOptions {
                    min_volume_m3: 1.0e-18,
                    min_scaled_jacobian: 0.05,
                    min_scaled_jacobian_improvement: 1.0e-12,
                    max_aspect_ratio_growth: 10.0,
                },
                max_attempted_points: 4,
                max_accepted_points: 1,
                relaxation: 0.5,
                max_projection_distance_m: 1.0,
            },
        );

        assert_eq!(report.attempted_point_count, 1);
        assert_eq!(report.accepted_point_count, 1);
        assert_eq!(report.rejected_point_count, 0);
        assert!(mesh.quality_optimized);
        assert_ne!(mesh.nodes[4].coordinates_m, [0.02, 0.02, 0.0]);
        assert_eq!(mesh.nodes[4].coordinates_m[2], 0.0);
        assert_eq!(
            mesh.evidence.entity_counts[TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ATTEMPT_COUNT],
            1
        );
        assert_eq!(
            mesh.evidence.entity_counts[TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ACCEPTED_COUNT],
            1
        );
    }

    #[test]
    fn boundary_smoothing_records_projection_rejection() {
        let mut mesh =
            boundary_smoothing_fixture([0.02, 0.02, 0.0], [Some(entity("edge_0")), None, None]);
        let projector = OutOfBoundsProjector;

        let report = smooth_tetrahedron_mesh_boundary_with_projector(
            &mut mesh,
            &projector,
            TetrahedronMeshBoundarySmoothingOptions {
                smoothing: TetrahedronSmoothingOptions {
                    min_scaled_jacobian: 0.15,
                    max_aspect_ratio_growth: 10.0,
                    ..TetrahedronSmoothingOptions::default()
                },
                max_attempted_points: 4,
                max_accepted_points: 1,
                relaxation: 0.5,
                max_projection_distance_m: 1.0,
            },
        );

        assert_eq!(report.attempted_point_count, 1);
        assert_eq!(report.accepted_point_count, 0);
        assert_eq!(report.rejected_point_count, 1);
        assert_eq!(
            mesh.evidence.entity_counts[TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_REJECTED_COUNT],
            1
        );
        assert_eq!(
            mesh.evidence.rejection_counts[&format!(
                "{TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_REJECTION_PREFIX}projection_out_of_bounds"
            )],
            1
        );
    }

    #[test]
    fn boundary_smoothing_skips_protected_edge_nodes() {
        let mut mesh = boundary_smoothing_fixture(
            [0.02, 0.02, 0.0],
            [None, Some(entity("edge_1")), Some(entity("edge_2"))],
        );
        let projector = PlanarProjector;

        let report = smooth_tetrahedron_mesh_boundary_with_projector(
            &mut mesh,
            &projector,
            TetrahedronMeshBoundarySmoothingOptions {
                smoothing: TetrahedronSmoothingOptions {
                    min_scaled_jacobian: 0.95,
                    max_aspect_ratio_growth: 10.0,
                    ..TetrahedronSmoothingOptions::default()
                },
                max_attempted_points: 4,
                max_accepted_points: 1,
                relaxation: 0.5,
                max_projection_distance_m: 1.0,
            },
        );

        assert_eq!(report.attempted_point_count, 0);
        assert!(!mesh.quality_optimized);
    }

    struct PlanarProjector;

    impl TetrahedronBoundarySmoothingProjector for PlanarProjector {
        fn project_to_source_face(
            &self,
            _source_face_id: &TopologyEntityId,
            point_m: Point3,
        ) -> Option<TetrahedronBoundarySmoothingProjection> {
            Some(TetrahedronBoundarySmoothingProjection {
                point_m: [point_m[0], point_m[1], 0.0],
                distance_m: point_m[2].abs(),
                in_bounds: true,
            })
        }
    }

    struct OutOfBoundsProjector;

    impl TetrahedronBoundarySmoothingProjector for OutOfBoundsProjector {
        fn project_to_source_face(
            &self,
            _source_face_id: &TopologyEntityId,
            point_m: Point3,
        ) -> Option<TetrahedronBoundarySmoothingProjection> {
            Some(TetrahedronBoundarySmoothingProjection {
                point_m: [point_m[0], point_m[1], 0.0],
                distance_m: point_m[2].abs(),
                in_bounds: false,
            })
        }
    }

    fn boundary_smoothing_fixture(
        boundary_point: Point3,
        source_edge_ids: [Option<TopologyEntityId>; 3],
    ) -> TetrahedronMesh {
        TetrahedronMesh {
            mesh_id: "boundary_smoothing_fixture".to_string(),
            tetrahedron_generation_family: "unknown".to_string(),
            nodes: vec![
                node("0", [0.0, 0.0, 0.0]),
                node("1", [1.0, 0.0, 0.0]),
                node("2", [0.0, 1.0, 0.0]),
                node("3", [0.0, 0.0, 1.0]),
                node("4", boundary_point),
            ],
            elements: vec![
                element("0", ["4", "0", "1", "3"]),
                element("1", ["4", "0", "2", "3"]),
            ],
            boundary_faces: vec![TetrahedronBoundaryFace {
                face_id: entity("boundary_0"),
                node_ids: [entity("0"), entity("1"), entity("4")],
                source_face_id: entity("face_0"),
                source_edge_ids,
            }],
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
