use super::*;

mod child_faces;
pub(in super::super) use child_faces::{
    edge_split_child_boundary_face, split_child_boundary_face, three_edge_split_child_boundary_face,
};

pub(in super::super) fn split_constrained_cavity_boundary_face_on_edge(
    face: &ConstrainedCavityBoundaryFace,
    edge: [u32; 2],
    split_node_id: u32,
) -> Result<[ConstrainedCavityBoundaryFace; 2], ConstrainedCavityBoundarySplitError> {
    if face.node_ids.contains(&split_node_id) {
        return Err(
            ConstrainedCavityBoundarySplitError::SplitNodeReusesFaceNode {
                node_id: split_node_id,
            },
        );
    }
    let sorted_split_edge = sorted_edge(edge);
    if !face_edges(face.node_ids)
        .into_iter()
        .any(|candidate| sorted_edge(candidate) == sorted_split_edge)
    {
        return Err(ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        });
    }
    let source_edges = boundary_face_source_edges(face).map_err(|_| {
        ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        }
    })?;
    let source_edge_id = source_edges.get(&sorted_split_edge).copied().flatten();
    let [a, b] = edge;
    let c = face
        .node_ids
        .into_iter()
        .find(|node_id| *node_id != a && *node_id != b)
        .ok_or(ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        })?;
    Ok([
        edge_split_child_boundary_face(
            face,
            [a, split_node_id, c],
            split_node_id,
            sorted_split_edge,
            source_edge_id,
            &source_edges,
        ),
        edge_split_child_boundary_face(
            face,
            [split_node_id, b, c],
            split_node_id,
            sorted_split_edge,
            source_edge_id,
            &source_edges,
        ),
    ])
}

pub(in super::super) fn split_constrained_cavity_boundary_faces_on_edge(
    faces: &[ConstrainedCavityBoundaryFace],
    face_node_ids: [u32; 3],
    edge: [u32; 2],
    split_node_id: u32,
) -> Result<Vec<ConstrainedCavityBoundaryFace>, ConstrainedCavityBoundarySplitError> {
    let target = sorted_face(face_node_ids);
    let split_edge = sorted_edge(edge);
    let mut split_faces = Vec::<ConstrainedCavityBoundaryFace>::with_capacity(faces.len() + 1);
    let mut found = false;
    for face in faces {
        if sorted_face(face.node_ids) == target {
            found = true;
        }
        if face_edges(face.node_ids)
            .into_iter()
            .any(|candidate| sorted_edge(candidate) == split_edge)
        {
            split_faces.extend(split_constrained_cavity_boundary_face_on_edge(
                face,
                edge,
                split_node_id,
            )?);
        } else {
            split_faces.push(face.clone());
        }
    }
    if !found {
        return Err(ConstrainedCavityBoundarySplitError::MissingBoundaryFace { node_ids: target });
    }
    Ok(split_faces)
}

pub(in super::super) fn split_constrained_cavity_boundary_faces_on_edge_patch(
    faces: &[ConstrainedCavityBoundaryFace],
    edge: [u32; 2],
    split_node_id: u32,
) -> Result<Vec<ConstrainedCavityBoundaryFace>, ConstrainedCavityBoundarySplitError> {
    let split_edge = sorted_edge(edge);
    let incident = faces
        .iter()
        .filter(|face| {
            face_edges(face.node_ids)
                .into_iter()
                .any(|candidate| sorted_edge(candidate) == split_edge)
        })
        .collect::<Vec<_>>();
    if incident.len() != 2 {
        return Err(ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: [split_edge[0], split_edge[1], split_node_id],
        });
    }
    let mut split_faces = Vec::<ConstrainedCavityBoundaryFace>::with_capacity(faces.len() + 2);
    for face in faces {
        if face_edges(face.node_ids)
            .into_iter()
            .any(|candidate| sorted_edge(candidate) == split_edge)
        {
            split_faces.extend(split_constrained_cavity_boundary_face_on_edge_patch(
                face,
                split_edge,
                split_node_id,
            )?);
        } else {
            split_faces.push(face.clone());
        }
    }
    Ok(split_faces)
}

pub(in super::super) fn split_constrained_cavity_boundary_face_on_edge_patch(
    face: &ConstrainedCavityBoundaryFace,
    split_edge: [u32; 2],
    split_node_id: u32,
) -> Result<[ConstrainedCavityBoundaryFace; 2], ConstrainedCavityBoundarySplitError> {
    if face.node_ids.contains(&split_node_id) {
        return Err(
            ConstrainedCavityBoundarySplitError::SplitNodeReusesFaceNode {
                node_id: split_node_id,
            },
        );
    }
    let opposite = face
        .node_ids
        .into_iter()
        .find(|node_id| !split_edge.contains(node_id))
        .ok_or(ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        })?;
    let perimeter_source_edges = boundary_face_source_edges(face).map_err(|_| {
        ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        }
    })?;
    Ok([
        split_child_boundary_face(
            face,
            [split_edge[0], opposite, split_node_id],
            &perimeter_source_edges,
        ),
        split_child_boundary_face(
            face,
            [opposite, split_edge[1], split_node_id],
            &perimeter_source_edges,
        ),
    ])
}

pub(in super::super) fn split_constrained_cavity_boundary_faces_on_three_edges(
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

pub(in super::super) fn split_constrained_cavity_boundary_face_on_three_edges(
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
