use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{
        Tetrahedron4Element, TetrahedronMesh, TopologyEntityId,
        TETRAHEDRON_EXACT_QUALITY_REPAIR_PASS_COUNT,
        TETRAHEDRON_EXACT_QUALITY_REPAIR_REJECTION_PREFIX,
        TETRAHEDRON_EXACT_QUALITY_SEED_STAR_RELOCATION_COUNT,
        TETRAHEDRON_EXACT_QUALITY_UNREPAIRED_INTERIOR_SEED_COUNT,
        TETRAHEDRON_EXACT_QUALITY_UNREPAIRED_TOTAL_COUNT,
    },
    quality::predicate::{tetrahedron_scaled_jacobian, Point3},
};
use runmat_meshing_opt::exact_quality::{
    evaluate_tetrahedron_exact_quality_repair_candidate, TetrahedronExactQuality,
    TetrahedronExactQualityRepairOptions,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TetrahedronMeshExactQualityRepairOptions {
    pub exact_quality: TetrahedronExactQualityRepairOptions,
    pub max_attempted_seeds: usize,
    pub max_relocated_seeds: usize,
    pub relaxation: f64,
}

impl Default for TetrahedronMeshExactQualityRepairOptions {
    fn default() -> Self {
        Self {
            exact_quality: TetrahedronExactQualityRepairOptions::default(),
            max_attempted_seeds: 16,
            max_relocated_seeds: 4,
            relaxation: 0.5,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TetrahedronMeshExactQualityRepairReport {
    pub attempted_seed_count: usize,
    pub relocated_seed_count: usize,
    pub rejected_seed_count: usize,
    pub unrepaired_seed_count: usize,
    pub rejected_by_reason: BTreeMap<String, usize>,
}

pub fn repair_tetrahedron_mesh_exact_quality(
    mesh: &mut TetrahedronMesh,
    options: TetrahedronMeshExactQualityRepairOptions,
) -> TetrahedronMeshExactQualityRepairReport {
    let mut report = TetrahedronMeshExactQualityRepairReport::default();
    if options.max_attempted_seeds == 0
        || options.max_relocated_seeds == 0
        || !options.relaxation.is_finite()
        || options.relaxation <= 0.0
        || mesh.elements.is_empty()
    {
        report.unrepaired_seed_count =
            exact_quality_violation_indices(&exact_quality_set(mesh), options.exact_quality).len();
        record_exact_quality_repair_evidence(mesh, &report);
        return report;
    }

    let boundary_nodes = boundary_node_ids(mesh);
    let mut relocated = 0_usize;

    loop {
        let current_quality = exact_quality_set(mesh);
        let seed_indices = exact_quality_violation_indices(&current_quality, options.exact_quality);
        if seed_indices.is_empty() {
            break;
        }
        if report.attempted_seed_count >= options.max_attempted_seeds {
            report.unrepaired_seed_count += seed_indices.len();
            break;
        }
        if relocated >= options.max_relocated_seeds {
            report.unrepaired_seed_count += seed_indices.len();
            break;
        }

        report.attempted_seed_count += 1;
        let Some(edit) =
            accepted_exact_quality_node_relocation(mesh, seed_indices[0], &boundary_nodes, options)
        else {
            report.unrepaired_seed_count += 1;
            record_exact_quality_repair_rejection(&mut report, "no_accepted_relocation");
            break;
        };
        if let Some(node) = mesh
            .nodes
            .iter_mut()
            .find(|node| node.node_id == edit.node_id)
        {
            node.coordinates_m = edit.coordinates_m;
            report.relocated_seed_count += 1;
            relocated += 1;
        } else {
            report.unrepaired_seed_count += 1;
            record_exact_quality_repair_rejection(&mut report, "missing_node");
            break;
        }
    }

    if report.attempted_seed_count > 0 {
        mesh.quality_optimized = true;
    }
    record_exact_quality_repair_evidence(mesh, &report);
    report
}

#[derive(Debug, Clone, PartialEq)]
struct ExactQualityNodeRelocation {
    node_id: TopologyEntityId,
    coordinates_m: Point3,
}

fn accepted_exact_quality_node_relocation(
    mesh: &TetrahedronMesh,
    element_index: usize,
    boundary_nodes: &BTreeSet<TopologyEntityId>,
    options: TetrahedronMeshExactQualityRepairOptions,
) -> Option<ExactQualityNodeRelocation> {
    let element = mesh.elements.get(element_index)?;
    let node_coordinates = node_coordinate_map(mesh);
    let incident_elements = incident_elements_by_node(mesh);
    let current_quality = exact_quality_set(mesh);

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
        let proposed_quality = exact_quality_set_with_relocated_node(mesh, node_id, candidate);
        let Ok(evaluation) = evaluate_tetrahedron_exact_quality_repair_candidate(
            &current_quality,
            &proposed_quality,
            options.exact_quality,
        ) else {
            continue;
        };
        if evaluation.accepted {
            return Some(ExactQualityNodeRelocation {
                node_id: node_id.clone(),
                coordinates_m: candidate,
            });
        }
    }

    None
}

fn exact_quality_violation_indices(
    quality: &[TetrahedronExactQuality],
    options: TetrahedronExactQualityRepairOptions,
) -> Vec<usize> {
    quality
        .iter()
        .filter(|quality| quality.exact_scaled_jacobian < options.min_exact_scaled_jacobian)
        .map(|quality| quality.tetrahedron_id as usize)
        .collect()
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
    options: TetrahedronMeshExactQualityRepairOptions,
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

fn exact_quality_set(mesh: &TetrahedronMesh) -> Vec<TetrahedronExactQuality> {
    mesh.elements
        .iter()
        .enumerate()
        .map(|(index, element)| exact_quality(index, element, mesh))
        .collect()
}

fn exact_quality_set_with_relocated_node(
    mesh: &TetrahedronMesh,
    relocated_node_id: &TopologyEntityId,
    relocated_coordinates: Point3,
) -> Vec<TetrahedronExactQuality> {
    mesh.elements
        .iter()
        .enumerate()
        .map(|(index, element)| {
            exact_quality_with_relocated_node(
                index,
                element,
                mesh,
                relocated_node_id,
                relocated_coordinates,
            )
        })
        .collect()
}

fn exact_quality(
    element_index: usize,
    element: &Tetrahedron4Element,
    mesh: &TetrahedronMesh,
) -> TetrahedronExactQuality {
    TetrahedronExactQuality {
        tetrahedron_id: element_index as u32,
        exact_scaled_jacobian: tetrahedron_scaled_jacobian(element_points(element, mesh)),
    }
}

fn exact_quality_with_relocated_node(
    element_index: usize,
    element: &Tetrahedron4Element,
    mesh: &TetrahedronMesh,
    relocated_node_id: &TopologyEntityId,
    relocated_coordinates: Point3,
) -> TetrahedronExactQuality {
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
    TetrahedronExactQuality {
        tetrahedron_id: element_index as u32,
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

fn record_exact_quality_repair_rejection(
    report: &mut TetrahedronMeshExactQualityRepairReport,
    reason: &str,
) {
    report.rejected_seed_count += 1;
    *report
        .rejected_by_reason
        .entry(reason.to_string())
        .or_default() += 1;
}

fn record_exact_quality_repair_evidence(
    mesh: &mut TetrahedronMesh,
    report: &TetrahedronMeshExactQualityRepairReport,
) {
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_EXACT_QUALITY_REPAIR_PASS_COUNT.to_string())
        .or_default() += usize::from(report.attempted_seed_count > 0);
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_EXACT_QUALITY_SEED_STAR_RELOCATION_COUNT.to_string())
        .or_default() += report.relocated_seed_count;
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_EXACT_QUALITY_UNREPAIRED_TOTAL_COUNT.to_string())
        .or_default() += report.unrepaired_seed_count;
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_EXACT_QUALITY_UNREPAIRED_INTERIOR_SEED_COUNT.to_string())
        .or_default() += report.unrepaired_seed_count;
    for (reason, count) in &report.rejected_by_reason {
        *mesh
            .evidence
            .rejection_counts
            .entry(format!(
                "{TETRAHEDRON_EXACT_QUALITY_REPAIR_REJECTION_PREFIX}{reason}"
            ))
            .or_default() += count;
    }
}

#[cfg(test)]
mod tests;
