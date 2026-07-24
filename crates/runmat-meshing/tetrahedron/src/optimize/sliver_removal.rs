use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{
        Tetrahedron4Element, TetrahedronMesh, TopologyEntityId,
        TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_ACCEPTED_COUNT,
        TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_ATTEMPT_COUNT,
        TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_BUDGET_LIMIT_COUNT,
        TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_REJECTED_COUNT,
        TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_REJECTION_PREFIX,
    },
    quality::predicate::{tetrahedron_edge_aspect_ratio, tetrahedron_scaled_jacobian, Point3},
};
use runmat_meshing_opt::sliver::{
    classify_sliver_tetrahedra, evaluate_sliver_removal, SliverRecoveryOptions,
    SliverTetrahedronQuality,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TetrahedronMeshSliverRemovalOptions {
    pub sliver: SliverRecoveryOptions,
    pub max_attempted_elements: usize,
    pub max_accepted_edits: usize,
    pub relaxation: f64,
}

impl Default for TetrahedronMeshSliverRemovalOptions {
    fn default() -> Self {
        Self {
            sliver: SliverRecoveryOptions::default(),
            max_attempted_elements: 16,
            max_accepted_edits: 4,
            relaxation: 0.5,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TetrahedronMeshSliverRemovalReport {
    pub attempted_element_count: usize,
    pub accepted_edit_count: usize,
    pub rejected_element_count: usize,
    pub budget_limited_element_count: usize,
    pub rejected_by_reason: BTreeMap<String, usize>,
}

pub fn remove_tetrahedron_mesh_slivers(
    mesh: &mut TetrahedronMesh,
    options: TetrahedronMeshSliverRemovalOptions,
) -> TetrahedronMeshSliverRemovalReport {
    let mut report = TetrahedronMeshSliverRemovalReport::default();
    if options.max_attempted_elements == 0
        || options.max_accepted_edits == 0
        || !options.relaxation.is_finite()
        || options.relaxation <= 0.0
        || mesh.elements.is_empty()
    {
        let quality = sliver_quality_set(mesh);
        if let Ok(slivers) = classify_sliver_tetrahedra(&quality, options.sliver) {
            report.budget_limited_element_count = slivers.len();
        }
        record_sliver_removal_evidence(mesh, &report);
        return report;
    }

    let boundary_nodes = boundary_node_ids(mesh);
    let mut accepted = 0_usize;

    loop {
        let current_quality = sliver_quality_set(mesh);
        let Ok(slivers) = classify_sliver_tetrahedra(&current_quality, options.sliver) else {
            record_sliver_removal_rejection(&mut report, "invalid_quality");
            break;
        };
        if slivers.is_empty() {
            break;
        }
        if report.attempted_element_count >= options.max_attempted_elements {
            report.budget_limited_element_count += slivers.len();
            break;
        }
        if accepted >= options.max_accepted_edits {
            report.budget_limited_element_count += slivers.len();
            break;
        }

        let element_index = slivers[0].tetrahedron_id as usize;
        report.attempted_element_count += 1;
        let Some(edit) =
            accepted_sliver_node_relocation(mesh, element_index, &boundary_nodes, options)
        else {
            record_sliver_removal_rejection(&mut report, "no_accepted_relocation");
            break;
        };

        if let Some(node) = mesh
            .nodes
            .iter_mut()
            .find(|node| node.node_id == edit.node_id)
        {
            node.coordinates_m = edit.coordinates_m;
            report.accepted_edit_count += 1;
            accepted += 1;
        } else {
            record_sliver_removal_rejection(&mut report, "missing_node");
            break;
        }
    }

    if report.attempted_element_count > 0 {
        mesh.quality_optimized = true;
    }
    record_sliver_removal_evidence(mesh, &report);
    report
}

#[derive(Debug, Clone, PartialEq)]
struct SliverNodeRelocation {
    node_id: TopologyEntityId,
    coordinates_m: Point3,
}

fn accepted_sliver_node_relocation(
    mesh: &TetrahedronMesh,
    element_index: usize,
    boundary_nodes: &BTreeSet<TopologyEntityId>,
    options: TetrahedronMeshSliverRemovalOptions,
) -> Option<SliverNodeRelocation> {
    let element = mesh.elements.get(element_index)?;
    let node_coordinates = node_coordinate_map(mesh);
    let incident_elements = incident_elements_by_node(mesh);
    let current_quality = sliver_quality_set(mesh);

    for node_id in element
        .node_ids
        .iter()
        .filter(|node_id| !boundary_nodes.contains(*node_id))
    {
        let Some(element_indices) = incident_elements.get(node_id) else {
            continue;
        };
        let Some(candidate) =
            smoothed_node_coordinates(node_id, element_indices, mesh, &node_coordinates, options)
        else {
            continue;
        };
        let proposed_quality = sliver_quality_set_with_relocated_node(mesh, node_id, candidate);
        let Ok(evaluation) =
            evaluate_sliver_removal(&current_quality, &proposed_quality, options.sliver)
        else {
            continue;
        };
        if evaluation.accepted {
            return Some(SliverNodeRelocation {
                node_id: node_id.clone(),
                coordinates_m: candidate,
            });
        }
    }

    None
}

fn boundary_node_ids(mesh: &TetrahedronMesh) -> BTreeSet<TopologyEntityId> {
    mesh.boundary_faces
        .iter()
        .flat_map(|face| face.node_ids.iter().cloned())
        .collect()
}

fn node_coordinate_map(mesh: &TetrahedronMesh) -> BTreeMap<TopologyEntityId, Point3> {
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
    node_id: &TopologyEntityId,
    element_indices: &[usize],
    mesh: &TetrahedronMesh,
    node_coordinates: &BTreeMap<TopologyEntityId, Point3>,
    options: TetrahedronMeshSliverRemovalOptions,
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

fn sliver_quality_set(mesh: &TetrahedronMesh) -> Vec<SliverTetrahedronQuality> {
    mesh.elements
        .iter()
        .enumerate()
        .map(|(index, element)| sliver_quality(index, element, mesh))
        .collect()
}

fn sliver_quality_set_with_relocated_node(
    mesh: &TetrahedronMesh,
    relocated_node_id: &TopologyEntityId,
    relocated_coordinates: Point3,
) -> Vec<SliverTetrahedronQuality> {
    mesh.elements
        .iter()
        .enumerate()
        .map(|(index, element)| {
            sliver_quality_with_relocated_node(
                index,
                element,
                mesh,
                relocated_node_id,
                relocated_coordinates,
            )
        })
        .collect()
}

fn sliver_quality(
    element_index: usize,
    element: &Tetrahedron4Element,
    mesh: &TetrahedronMesh,
) -> SliverTetrahedronQuality {
    let points = element_points(element, mesh);
    SliverTetrahedronQuality {
        tetrahedron_id: element_index as u32,
        aspect_ratio: tetrahedron_edge_aspect_ratio(points),
        exact_scaled_jacobian: tetrahedron_scaled_jacobian(points),
    }
}

fn sliver_quality_with_relocated_node(
    element_index: usize,
    element: &Tetrahedron4Element,
    mesh: &TetrahedronMesh,
    relocated_node_id: &TopologyEntityId,
    relocated_coordinates: Point3,
) -> SliverTetrahedronQuality {
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
    SliverTetrahedronQuality {
        tetrahedron_id: element_index as u32,
        aspect_ratio: tetrahedron_edge_aspect_ratio(points),
        exact_scaled_jacobian: tetrahedron_scaled_jacobian(points),
    }
}

fn element_points(element: &Tetrahedron4Element, mesh: &TetrahedronMesh) -> [Point3; 4] {
    element.node_ids.clone().map(|node_id| {
        mesh.nodes
            .iter()
            .find(|node| node.node_id == node_id)
            .map(|node| node.coordinates_m)
            .unwrap_or([f64::NAN, f64::NAN, f64::NAN])
    })
}

fn record_sliver_removal_rejection(report: &mut TetrahedronMeshSliverRemovalReport, reason: &str) {
    report.rejected_element_count += 1;
    *report
        .rejected_by_reason
        .entry(reason.to_string())
        .or_default() += 1;
}

fn record_sliver_removal_evidence(
    mesh: &mut TetrahedronMesh,
    report: &TetrahedronMeshSliverRemovalReport,
) {
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_ATTEMPT_COUNT.to_string())
        .or_default() += report.attempted_element_count;
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_ACCEPTED_COUNT.to_string())
        .or_default() += report.accepted_edit_count;
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_REJECTED_COUNT.to_string())
        .or_default() += report.rejected_element_count;
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_BUDGET_LIMIT_COUNT.to_string())
        .or_default() += report.budget_limited_element_count;
    for (reason, count) in &report.rejected_by_reason {
        *mesh
            .evidence
            .rejection_counts
            .entry(format!(
                "{TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_REJECTION_PREFIX}{reason}"
            ))
            .or_default() += count;
    }
}

#[cfg(test)]
mod tests;
