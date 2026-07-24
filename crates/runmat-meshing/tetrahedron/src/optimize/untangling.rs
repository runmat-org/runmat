use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{
        Tetrahedron4Element, TetrahedronMesh, TopologyEntityId,
        TETRAHEDRON_UNTANGLING_FINAL_NEAR_SINGULAR_COUNT,
        TETRAHEDRON_UNTANGLING_INITIAL_NEAR_SINGULAR_COUNT, TETRAHEDRON_UNTANGLING_PASS_COUNT,
        TETRAHEDRON_UNTANGLING_REJECTION_PREFIX, TETRAHEDRON_UNTANGLING_RELOCATED_SEED_COUNT,
    },
    quality::predicate::{tetrahedron_scaled_jacobian, Point3},
};
use runmat_meshing_opt::untangle::{
    evaluate_tetrahedron_untangling_candidate, TetrahedronUntanglingOptions,
    TetrahedronUntanglingQuality,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TetrahedronMeshUntanglingOptions {
    pub untangling: TetrahedronUntanglingOptions,
    pub max_attempted_seeds: usize,
    pub max_relocated_seeds: usize,
    pub relaxation: f64,
}

impl Default for TetrahedronMeshUntanglingOptions {
    fn default() -> Self {
        Self {
            untangling: TetrahedronUntanglingOptions::default(),
            max_attempted_seeds: 16,
            max_relocated_seeds: 4,
            relaxation: 0.5,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TetrahedronMeshUntanglingReport {
    pub initial_near_singular_count: usize,
    pub final_near_singular_count: usize,
    pub attempted_seed_count: usize,
    pub relocated_seed_count: usize,
    pub rejected_seed_count: usize,
    pub rejected_by_reason: BTreeMap<String, usize>,
}

pub fn untangle_tetrahedron_mesh_interior(
    mesh: &mut TetrahedronMesh,
    options: TetrahedronMeshUntanglingOptions,
) -> TetrahedronMeshUntanglingReport {
    let initial_quality = untangling_quality_set(mesh);
    let mut report = TetrahedronMeshUntanglingReport {
        initial_near_singular_count: near_singular_count(&initial_quality, options.untangling),
        final_near_singular_count: 0,
        ..TetrahedronMeshUntanglingReport::default()
    };
    if options.max_attempted_seeds == 0
        || options.max_relocated_seeds == 0
        || !options.relaxation.is_finite()
        || options.relaxation <= 0.0
        || mesh.elements.is_empty()
    {
        report.final_near_singular_count = report.initial_near_singular_count;
        record_untangling_evidence(mesh, &report);
        return report;
    }

    let boundary_nodes = boundary_node_ids(mesh);
    let mut relocated = 0_usize;

    loop {
        let current_quality = untangling_quality_set(mesh);
        let seed_indices = near_singular_element_indices(&current_quality, options.untangling);
        if seed_indices.is_empty() {
            break;
        }
        if report.attempted_seed_count >= options.max_attempted_seeds {
            break;
        }
        if relocated >= options.max_relocated_seeds {
            break;
        }

        report.attempted_seed_count += 1;
        let Some(edit) =
            accepted_untangling_node_relocation(mesh, seed_indices[0], &boundary_nodes, options)
        else {
            record_untangling_rejection(&mut report, "no_accepted_relocation");
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
            record_untangling_rejection(&mut report, "missing_node");
            break;
        }
    }

    report.final_near_singular_count =
        near_singular_count(&untangling_quality_set(mesh), options.untangling);
    if report.attempted_seed_count > 0 {
        mesh.quality_optimized = true;
    }
    record_untangling_evidence(mesh, &report);
    report
}

#[derive(Debug, Clone, PartialEq)]
struct UntanglingNodeRelocation {
    node_id: TopologyEntityId,
    coordinates_m: Point3,
}

fn accepted_untangling_node_relocation(
    mesh: &TetrahedronMesh,
    element_index: usize,
    boundary_nodes: &BTreeSet<TopologyEntityId>,
    options: TetrahedronMeshUntanglingOptions,
) -> Option<UntanglingNodeRelocation> {
    let element = mesh.elements.get(element_index)?;
    let node_coordinates = node_coordinate_map(mesh);
    let incident_elements = incident_elements_by_node(mesh);
    let current_quality = untangling_quality_set(mesh);

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
        let proposed_quality = untangling_quality_set_with_relocated_node(mesh, node_id, candidate);
        let Ok(evaluation) = evaluate_tetrahedron_untangling_candidate(
            &current_quality,
            &proposed_quality,
            options.untangling,
        ) else {
            continue;
        };
        if evaluation.accepted {
            return Some(UntanglingNodeRelocation {
                node_id: node_id.clone(),
                coordinates_m: candidate,
            });
        }
    }

    None
}

fn near_singular_element_indices(
    quality: &[TetrahedronUntanglingQuality],
    options: TetrahedronUntanglingOptions,
) -> Vec<usize> {
    quality
        .iter()
        .filter(|quality| quality.scaled_jacobian < options.near_singular_scaled_jacobian)
        .map(|quality| quality.tetrahedron_id as usize)
        .collect()
}

fn near_singular_count(
    quality: &[TetrahedronUntanglingQuality],
    options: TetrahedronUntanglingOptions,
) -> usize {
    near_singular_element_indices(quality, options).len()
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
    options: TetrahedronMeshUntanglingOptions,
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

fn untangling_quality_set(mesh: &TetrahedronMesh) -> Vec<TetrahedronUntanglingQuality> {
    mesh.elements
        .iter()
        .enumerate()
        .map(|(index, element)| untangling_quality(index, element, mesh))
        .collect()
}

fn untangling_quality_set_with_relocated_node(
    mesh: &TetrahedronMesh,
    relocated_node_id: &TopologyEntityId,
    relocated_coordinates: Point3,
) -> Vec<TetrahedronUntanglingQuality> {
    mesh.elements
        .iter()
        .enumerate()
        .map(|(index, element)| {
            untangling_quality_with_relocated_node(
                index,
                element,
                mesh,
                relocated_node_id,
                relocated_coordinates,
            )
        })
        .collect()
}

fn untangling_quality(
    element_index: usize,
    element: &Tetrahedron4Element,
    mesh: &TetrahedronMesh,
) -> TetrahedronUntanglingQuality {
    TetrahedronUntanglingQuality {
        tetrahedron_id: element_index as u32,
        scaled_jacobian: tetrahedron_scaled_jacobian(element_points(element, mesh)),
    }
}

fn untangling_quality_with_relocated_node(
    element_index: usize,
    element: &Tetrahedron4Element,
    mesh: &TetrahedronMesh,
    relocated_node_id: &TopologyEntityId,
    relocated_coordinates: Point3,
) -> TetrahedronUntanglingQuality {
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
    TetrahedronUntanglingQuality {
        tetrahedron_id: element_index as u32,
        scaled_jacobian: tetrahedron_scaled_jacobian(points),
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

fn record_untangling_rejection(report: &mut TetrahedronMeshUntanglingReport, reason: &str) {
    report.rejected_seed_count += 1;
    *report
        .rejected_by_reason
        .entry(reason.to_string())
        .or_default() += 1;
}

fn record_untangling_evidence(
    mesh: &mut TetrahedronMesh,
    report: &TetrahedronMeshUntanglingReport,
) {
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_UNTANGLING_INITIAL_NEAR_SINGULAR_COUNT.to_string())
        .or_default() += report.initial_near_singular_count;
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_UNTANGLING_FINAL_NEAR_SINGULAR_COUNT.to_string())
        .or_default() += report.final_near_singular_count;
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_UNTANGLING_PASS_COUNT.to_string())
        .or_default() += usize::from(report.attempted_seed_count > 0);
    *mesh
        .evidence
        .entity_counts
        .entry(TETRAHEDRON_UNTANGLING_RELOCATED_SEED_COUNT.to_string())
        .or_default() += report.relocated_seed_count;
    for (reason, count) in &report.rejected_by_reason {
        *mesh
            .evidence
            .rejection_counts
            .entry(format!("{TETRAHEDRON_UNTANGLING_REJECTION_PREFIX}{reason}"))
            .or_default() += count;
    }
}

#[cfg(test)]
mod tests;
