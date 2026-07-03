use super::*;

pub fn constrained_cavity_solid_empty_boundary_faces(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<Vec<[u32; 3]>, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    Ok(solid_empty_boundary_faces(
        cavity,
        &boundary_node_map,
        &boundary_triangles,
        options,
    ))
}

pub fn constrained_cavity_classified_solid_empty_boundary_faces(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<ConstrainedCavitySolidEmptyBoundaryFaces, ConstrainedCavityRefillError> {
    let faces = constrained_cavity_solid_empty_boundary_faces(cavity, nodes, options)?;
    let boundary_faces = boundary_face_map(&cavity.boundary_faces)
        .map_err(ConstrainedCavityRefillError::Validation)?;
    let mut true_exterior_faces = Vec::<[u32; 3]>::new();
    let mut expandable_faces = Vec::<[u32; 3]>::new();
    for face in &faces {
        let Some(boundary_face) = boundary_faces.get(face) else {
            continue;
        };
        if boundary_face.outside_tetrahedron_ids.is_empty() {
            true_exterior_faces.push(*face);
        } else {
            expandable_faces.push(*face);
        }
    }
    Ok(ConstrainedCavitySolidEmptyBoundaryFaces {
        faces,
        true_exterior_faces,
        expandable_faces,
    })
}

pub fn recover_constrained_cavity_solid_empty_boundaries(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    source_tetrahedra: &[CavityTetrahedron],
    source_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<
    ConstrainedCavitySolidEmptyBoundaryRecovery,
    ConstrainedCavitySolidEmptyBoundaryRecoveryError,
> {
    let classification =
        constrained_cavity_classified_solid_empty_boundary_faces(cavity, nodes, options)
            .map_err(ConstrainedCavitySolidEmptyBoundaryRecoveryError::Refill)?;
    let mut current_cavity = cavity.clone();
    let mut current_nodes = nodes.to_vec();
    let mut split_nodes = Vec::<ConstrainedCavityNode>::new();
    let mut split_steps = Vec::<ConstrainedCavityBoundaryPatchSplitStep>::new();
    let mut rejected_splits = Vec::<ConstrainedCavitySolidEmptyBoundaryRejectedSplit>::new();
    let mut expanded_removed_tetrahedron_ids = Vec::<u32>::new();
    if !classification.expandable_faces.is_empty() {
        let before = current_cavity
            .removed_tetrahedron_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        current_cavity = constrained_cavity_expanded_across_boundary_faces(
            &current_cavity,
            source_tetrahedra,
            &classification.expandable_faces,
        )
        .map_err(ConstrainedCavitySolidEmptyBoundaryRecoveryError::Expansion)?;
        expanded_removed_tetrahedron_ids = current_cavity
            .removed_tetrahedron_ids
            .iter()
            .copied()
            .filter(|tetrahedron_id| !before.contains(tetrahedron_id))
            .collect();
        current_nodes = constrained_cavity_boundary_nodes_from_sources(
            &current_cavity,
            &current_nodes,
            source_nodes,
        )
        .map_err(ConstrainedCavitySolidEmptyBoundaryRecoveryError::Refill)?;
    }

    let split_classification = constrained_cavity_classified_solid_empty_boundary_faces(
        &current_cavity,
        &current_nodes,
        options,
    )
    .map_err(ConstrainedCavitySolidEmptyBoundaryRecoveryError::Refill)?;
    if !split_classification.true_exterior_faces.is_empty() {
        let patch_split = split_constrained_cavity_boundary_patch_at_centroids(
            &current_cavity,
            &current_nodes,
            &[],
            &split_classification.true_exterior_faces,
        )
        .map_err(ConstrainedCavitySolidEmptyBoundaryRecoveryError::Split)?;
        let mut split_candidate_nodes = current_nodes.clone();
        split_candidate_nodes.extend(patch_split.split_nodes.clone());
        let split_candidate_classification =
            constrained_cavity_classified_solid_empty_boundary_faces(
                &patch_split.cavity,
                &split_candidate_nodes,
                options,
            )
            .map_err(ConstrainedCavitySolidEmptyBoundaryRecoveryError::Refill)?;
        if split_candidate_classification.faces.len() <= split_classification.faces.len() {
            current_cavity = patch_split.cavity;
            split_nodes = patch_split.split_nodes;
            split_steps = patch_split.steps;
        } else {
            rejected_splits.push(ConstrainedCavitySolidEmptyBoundaryRejectedSplit {
                input_faces: split_classification.true_exterior_faces,
                output_faces: split_candidate_classification.faces,
                split_node_count: patch_split.split_nodes.len(),
                split_step_count: patch_split.steps.len(),
            });
        }
    }

    Ok(ConstrainedCavitySolidEmptyBoundaryRecovery {
        cavity: current_cavity,
        split_nodes,
        classification,
        split_steps,
        rejected_splits,
        expanded_removed_tetrahedron_ids,
    })
}

fn constrained_cavity_boundary_nodes_from_sources(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    source_nodes: &[ConstrainedCavityNode],
) -> Result<Vec<ConstrainedCavityNode>, ConstrainedCavityRefillError> {
    let mut coordinates = source_nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    coordinates.extend(nodes.iter().map(|node| (node.node_id, node.coordinates_m)));
    cavity_boundary_node_ids(cavity)
        .into_iter()
        .map(|node_id| {
            coordinates
                .get(&node_id)
                .copied()
                .map(|coordinates_m| ConstrainedCavityNode {
                    node_id,
                    coordinates_m,
                })
                .ok_or(ConstrainedCavityRefillError::MissingBoundaryNode { node_id })
        })
        .collect()
}

pub(super) fn solid_empty_boundary_faces(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> Vec<[u32; 3]> {
    let boundary_node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut solid_faces = BTreeSet::<[u32; 3]>::new();
    for first in 0..boundary_node_ids.len() {
        for second in (first + 1)..boundary_node_ids.len() {
            for third in (second + 1)..boundary_node_ids.len() {
                for fourth in (third + 1)..boundary_node_ids.len() {
                    let tetrahedron_node_ids = [
                        boundary_node_ids[first],
                        boundary_node_ids[second],
                        boundary_node_ids[third],
                        boundary_node_ids[fourth],
                    ];
                    let candidate_faces = tetrahedron_faces(tetrahedron_node_ids).map(sorted_face);
                    if !candidate_faces
                        .iter()
                        .any(|face| boundary_faces.contains(face))
                    {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| boundary_nodes[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        options,
                    )
                    .is_ok()
                    {
                        for face in candidate_faces {
                            if boundary_faces.contains(&face) {
                                solid_faces.insert(face);
                            }
                        }
                    }
                }
            }
        }
    }
    boundary_faces
        .into_iter()
        .filter(|face| !solid_faces.contains(face))
        .collect()
}
