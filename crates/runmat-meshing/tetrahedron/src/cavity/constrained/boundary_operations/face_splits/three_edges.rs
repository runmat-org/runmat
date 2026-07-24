use super::*;

pub(in super::super::super) fn split_constrained_cavity_boundary_faces_on_three_edges(
    faces: &[ConstrainedCavityBoundaryFace],
    face_node_ids: [u32; 3],
    edge_split_node_ids: BTreeMap<[u32; 2], u32>,
) -> Result<Vec<ConstrainedCavityBoundaryFace>, ConstrainedCavityBoundarySplitError> {
    let target = sorted_face(face_node_ids);
    let mut split_faces = Vec::<ConstrainedCavityBoundaryFace>::with_capacity(faces.len() + 6);
    let mut found = false;
    for face in faces {
        if sorted_face(face.node_ids) == target {
            found = true;
            split_faces.extend(split_constrained_cavity_boundary_face_on_three_edges(
                face,
                &edge_split_node_ids,
            )?);
            continue;
        }
        let split_edges = face_edges(face.node_ids)
            .into_iter()
            .filter_map(|edge| {
                edge_split_node_ids
                    .get(&sorted_edge(edge))
                    .copied()
                    .map(|node_id| (edge, node_id))
            })
            .collect::<Vec<_>>();
        if split_edges.is_empty() {
            split_faces.push(face.clone());
            continue;
        }
        if split_edges.len() > 1 {
            return Err(ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
                node_ids: sorted_face(face.node_ids),
            });
        }
        let (edge, split_node_id) = split_edges[0];
        split_faces.extend(split_constrained_cavity_boundary_face_on_edge(
            face,
            edge,
            split_node_id,
        )?);
    }
    if !found {
        return Err(ConstrainedCavityBoundarySplitError::MissingBoundaryFace { node_ids: target });
    }
    Ok(split_faces)
}

pub(in super::super::super) fn split_constrained_cavity_boundary_face_on_three_edges(
    face: &ConstrainedCavityBoundaryFace,
    edge_split_node_ids: &BTreeMap<[u32; 2], u32>,
) -> Result<[ConstrainedCavityBoundaryFace; 4], ConstrainedCavityBoundarySplitError> {
    let [a, b, c] = face.node_ids;
    let ab = *edge_split_node_ids.get(&sorted_edge([a, b])).ok_or(
        ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        },
    )?;
    let bc = *edge_split_node_ids.get(&sorted_edge([b, c])).ok_or(
        ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        },
    )?;
    let ca = *edge_split_node_ids.get(&sorted_edge([c, a])).ok_or(
        ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        },
    )?;
    let perimeter_source_edges = boundary_face_source_edges(face).map_err(|_| {
        ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        }
    })?;
    Ok([
        three_edge_split_child_boundary_face(
            face,
            [a, ab, ca],
            &perimeter_source_edges,
            edge_split_node_ids,
        ),
        three_edge_split_child_boundary_face(
            face,
            [ab, b, bc],
            &perimeter_source_edges,
            edge_split_node_ids,
        ),
        three_edge_split_child_boundary_face(
            face,
            [ca, bc, c],
            &perimeter_source_edges,
            edge_split_node_ids,
        ),
        three_edge_split_child_boundary_face(
            face,
            [ab, bc, ca],
            &perimeter_source_edges,
            edge_split_node_ids,
        ),
    ])
}
