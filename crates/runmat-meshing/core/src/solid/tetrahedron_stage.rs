use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::{
    contracts::{MeshingStage, Tetrahedron4Element, TopologyEntityId},
    predicate::{distance_squared, tetrahedron_centroid, tetrahedron_volume},
    size::field::MeshSizingField,
    source_topology::SourceTopologyModel,
    surface::SurfaceDiscretization,
    tetrahedron::generate::TetrahedronMesh,
};

use super::requested_refinement_selection;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolidTetrahedronStageEvidence {
    pub volume_component_count: usize,
    pub interior_seed_point_count: usize,
    pub expected_volume_m3: f64,
    pub expected_boundary_area_m2: f64,
    #[serde(default)]
    pub coverage_sample_points_m: Vec<[f64; 3]>,
    pub recovered_component_ratio: f64,
    pub requested_refinement_point_count: usize,
    pub accepted_requested_refinement_point_count: usize,
    pub accepted_requested_refinement_surrogate_point_count: usize,
    pub rejected_requested_refinement_point_count: usize,
    #[serde(default)]
    pub requested_refinement_rejected_by_reason: BTreeMap<String, usize>,
    pub dropped_requested_refinement_point_count: usize,
    #[serde(default)]
    pub requested_refinement_dropped_by_reason: BTreeMap<String, usize>,
}

pub(super) fn build_solid_tetrahedron_stage_evidence(
    topology: &SourceTopologyModel,
    surface: &SurfaceDiscretization,
    tetrahedron_mesh: &TetrahedronMesh,
    sizing: Option<&MeshSizingField>,
) -> SolidTetrahedronStageEvidence {
    let requested = requested_refinement_selection(topology, sizing);
    let tetrahedron_nodes_by_id = tetrahedron_mesh
        .nodes
        .iter()
        .map(|node| (node.node_id.clone(), node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut expected_volume_m3 = 0.0;
    let mut coverage_sample_points_m = Vec::<[f64; 3]>::new();
    for tetrahedron in &tetrahedron_mesh.elements {
        let Some(points) = tetrahedron_points_from_mesh(tetrahedron, &tetrahedron_nodes_by_id)
        else {
            continue;
        };
        expected_volume_m3 += tetrahedron_volume(points).abs();
        coverage_sample_points_m.push(tetrahedron_centroid(points));
    }
    coverage_sample_points_m.extend(requested.points[..requested.count].iter().copied());
    coverage_sample_points_m.truncate(64);

    let accepted_requested_refinement_point_count = requested
        .points
        .iter()
        .take(requested.count)
        .filter(|point| tetrahedron_mesh_has_node_near(tetrahedron_mesh, **point))
        .count();
    let rejected_requested_refinement_point_count = requested
        .count
        .saturating_sub(accepted_requested_refinement_point_count);
    let mut requested_refinement_rejected_by_reason = BTreeMap::<String, usize>::new();
    if rejected_requested_refinement_point_count > 0 {
        requested_refinement_rejected_by_reason.insert(
            "native_tetrahedron_generator_has_no_requested_point_insertion".to_string(),
            rejected_requested_refinement_point_count,
        );
    }

    SolidTetrahedronStageEvidence {
        volume_component_count: usize::from(!tetrahedron_mesh.elements.is_empty()),
        interior_seed_point_count: tetrahedron_mesh
            .nodes
            .iter()
            .filter(|node| node.node_id.stage == MeshingStage::TetrahedronMesh)
            .count(),
        expected_volume_m3,
        expected_boundary_area_m2: surface.elements.iter().map(|element| element.area_m2).sum(),
        coverage_sample_points_m,
        recovered_component_ratio: 1.0,
        requested_refinement_point_count: requested.count,
        accepted_requested_refinement_point_count,
        accepted_requested_refinement_surrogate_point_count: 0,
        rejected_requested_refinement_point_count,
        requested_refinement_rejected_by_reason,
        dropped_requested_refinement_point_count: 0,
        requested_refinement_dropped_by_reason: BTreeMap::new(),
    }
}

fn tetrahedron_points_from_mesh(
    tetrahedron: &Tetrahedron4Element,
    nodes: &BTreeMap<TopologyEntityId, [f64; 3]>,
) -> Option<TetrahedronPoints> {
    Some([
        *nodes.get(&tetrahedron.node_ids[0])?,
        *nodes.get(&tetrahedron.node_ids[1])?,
        *nodes.get(&tetrahedron.node_ids[2])?,
        *nodes.get(&tetrahedron.node_ids[3])?,
    ])
}

type TetrahedronPoints = [[f64; 3]; 4];

pub(super) fn tetrahedron_mesh_has_node_near(
    tetrahedron_mesh: &TetrahedronMesh,
    point: [f64; 3],
) -> bool {
    tetrahedron_mesh
        .nodes
        .iter()
        .any(|node| distance_squared(node.coordinates_m, point) <= 1.0e-20)
}
