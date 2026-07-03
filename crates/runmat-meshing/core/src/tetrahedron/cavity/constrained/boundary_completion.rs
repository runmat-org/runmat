use std::collections::BTreeMap;

use crate::{
    predicate::{
        point_in_closed_triangle_surface, tetrahedron_centroid, Point3, PointInClosedSurface,
        Triangle3,
    },
    tolerance::MeshingTolerance,
};

use super::{
    boundary_splits::{
        boundary_face_edge_split_node_candidates, boundary_face_mid_edge_split_nodes,
        boundary_face_split_node_candidates, edge_split_completion_tetrahedra_for_node,
        split_completion_tetrahedra_for_node, three_edge_split_completion_tetrahedra_for_node,
    },
    cavity_boundary_node_ids, raw_refill_tetrahedron_with_rejection_reason,
    refill_faces::refill_boundary_face_delta,
    split_constrained_cavity_boundary_faces, split_constrained_cavity_boundary_faces_on_edge,
    split_constrained_cavity_boundary_faces_on_three_edges,
    topology::{face_edges, sorted_edge, sorted_face, sorted_tetrahedron_nodes},
    validate_constrained_cavity, ConstrainedCavity, ConstrainedCavityNode,
    ConstrainedCavityRefillOptions, ConstrainedCavityRefillTetrahedron,
    ConstrainedCavityValidationError,
};

pub(super) fn complete_missing_boundary_face_tetrahedra(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    mut refill_tetrahedra: Vec<ConstrainedCavityRefillTetrahedron>,
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> Result<
    Result<
        (
            ConstrainedCavity,
            Vec<ConstrainedCavityRefillTetrahedron>,
            Vec<ConstrainedCavityNode>,
        ),
        &'static str,
    >,
    ConstrainedCavityValidationError,
> {
    let mut refined_cavity = cavity.clone();
    let mut refined_boundary_nodes = boundary_nodes.clone();
    let mut inserted_nodes = Vec::<ConstrainedCavityNode>::new();
    let mut changed = false;
    loop {
        let boundary_delta = refill_boundary_face_delta(&refined_cavity, &refill_tetrahedra)?;
        if boundary_delta.missing.is_empty() {
            if boundary_delta.unexpected.is_empty() {
                break;
            }
            let Some((_, tetrahedron)) = best_boundary_face_completion_tetrahedron_for_faces(
                &boundary_delta.unexpected,
                &refined_cavity,
                &refined_boundary_nodes,
                &refill_tetrahedra,
                boundary_triangles,
                options,
            )?
            else {
                return Ok(Err("boundary_node_completion_no_candidate"));
            };
            refill_tetrahedra.push(tetrahedron);
            changed = true;
            continue;
        }
        if let Some((_, tetrahedron)) = best_boundary_face_completion_tetrahedron_for_faces(
            &boundary_delta.missing,
            &refined_cavity,
            &refined_boundary_nodes,
            &refill_tetrahedra,
            boundary_triangles,
            options,
        )? {
            refill_tetrahedra.push(tetrahedron);
            changed = true;
            continue;
        }

        let split_completion = if let Some((split_cavity, split_node, split_tetrahedra)) =
            best_boundary_face_edge_split_completion_for_faces(
                &boundary_delta.missing,
                &refined_cavity,
                &refined_boundary_nodes,
                boundary_triangles,
                &refill_tetrahedra,
                options,
            )? {
            Some((split_cavity, vec![split_node], split_tetrahedra))
        } else if let Some((split_cavity, split_node, split_tetrahedra)) =
            best_boundary_face_split_completion_for_faces(
                &boundary_delta.missing,
                &refined_cavity,
                &refined_boundary_nodes,
                boundary_triangles,
                &refill_tetrahedra,
                options,
            )?
        {
            Some((split_cavity, vec![split_node], split_tetrahedra))
        } else {
            best_boundary_face_three_edge_split_completion_for_faces(
                &boundary_delta.missing,
                &refined_cavity,
                &refined_boundary_nodes,
                boundary_triangles,
                &refill_tetrahedra,
                options,
            )?
        };
        let Some((split_cavity, split_nodes, split_tetrahedra)) = split_completion else {
            return Ok(Err("boundary_node_completion_no_candidate"));
        };
        for split_node in split_nodes {
            refined_boundary_nodes.insert(split_node.node_id, split_node.coordinates_m);
            inserted_nodes.push(split_node);
        }
        refined_cavity = split_cavity;
        refill_tetrahedra.extend(split_tetrahedra);
        changed = true;
    }
    if changed {
        Ok(Ok((refined_cavity, refill_tetrahedra, inserted_nodes)))
    } else {
        Ok(Err("boundary_node_completion_no_missing_faces"))
    }
}

pub(super) fn best_boundary_face_completion_tetrahedron_for_faces(
    faces: &[[u32; 3]],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    refill_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> Result<Option<([u32; 3], ConstrainedCavityRefillTetrahedron)>, ConstrainedCavityValidationError>
{
    let current_delta = refill_boundary_face_delta(cavity, refill_tetrahedra)?;
    let current_delta_count = current_delta.missing.len() + current_delta.unexpected.len();
    let mut best = None::<([u32; 3], ConstrainedCavityRefillTetrahedron, usize)>;
    for face in faces {
        let Some(tetrahedron) = best_boundary_face_completion_tetrahedron(
            *face,
            cavity,
            boundary_nodes,
            refill_tetrahedra,
            boundary_triangles,
            options,
        ) else {
            continue;
        };
        let mut candidate_tetrahedra = refill_tetrahedra.to_vec();
        candidate_tetrahedra.push(tetrahedron.clone());
        let candidate_delta = refill_boundary_face_delta(cavity, &candidate_tetrahedra)?;
        let candidate_delta_count =
            candidate_delta.missing.len() + candidate_delta.unexpected.len();
        if candidate_delta_count >= current_delta_count {
            continue;
        }
        if best
            .as_ref()
            .is_none_or(|(_, best_tetrahedron, best_delta)| {
                candidate_delta_count < *best_delta
                    || (candidate_delta_count == *best_delta
                        && tetrahedron.exact_scaled_jacobian
                            > best_tetrahedron.exact_scaled_jacobian)
            })
        {
            best = Some((*face, tetrahedron, candidate_delta_count));
        }
    }
    Ok(best.map(|(face, tetrahedron, _)| (face, tetrahedron)))
}

pub(super) fn best_boundary_face_split_completion_for_faces(
    faces: &[[u32; 3]],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    refill_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    options: ConstrainedCavityRefillOptions,
) -> Result<
    Option<(
        ConstrainedCavity,
        ConstrainedCavityNode,
        Vec<ConstrainedCavityRefillTetrahedron>,
    )>,
    ConstrainedCavityValidationError,
> {
    let current_delta = refill_boundary_face_delta(cavity, refill_tetrahedra)?;
    let current_delta_count = current_delta.missing.len() + current_delta.unexpected.len();
    let mut best = None::<(
        ConstrainedCavity,
        ConstrainedCavityNode,
        Vec<ConstrainedCavityRefillTetrahedron>,
        f64,
    )>;
    for face in faces {
        let Some((split_cavity, split_node, split_tetrahedra)) =
            best_boundary_face_split_completion(
                *face,
                cavity,
                boundary_nodes,
                boundary_triangles,
                refill_tetrahedra,
                options,
            )?
        else {
            continue;
        };
        let mut candidate_tetrahedra = refill_tetrahedra.to_vec();
        candidate_tetrahedra.extend(split_tetrahedra.clone());
        let candidate_delta = refill_boundary_face_delta(&split_cavity, &candidate_tetrahedra)?;
        let candidate_delta_count =
            candidate_delta.missing.len() + candidate_delta.unexpected.len();
        if candidate_delta_count >= current_delta_count {
            continue;
        }
        let min_quality = split_tetrahedra
            .iter()
            .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min);
        if best
            .as_ref()
            .is_none_or(|(_, _, _, best_quality)| min_quality > *best_quality)
        {
            best = Some((split_cavity, split_node, split_tetrahedra, min_quality));
        }
    }
    Ok(best.map(|(split_cavity, split_node, split_tetrahedra, _)| {
        (split_cavity, split_node, split_tetrahedra)
    }))
}

pub(super) fn best_boundary_face_edge_split_completion_for_faces(
    faces: &[[u32; 3]],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    refill_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    options: ConstrainedCavityRefillOptions,
) -> Result<
    Option<(
        ConstrainedCavity,
        ConstrainedCavityNode,
        Vec<ConstrainedCavityRefillTetrahedron>,
    )>,
    ConstrainedCavityValidationError,
> {
    let current_delta = refill_boundary_face_delta(cavity, refill_tetrahedra)?;
    let current_delta_count = current_delta.missing.len() + current_delta.unexpected.len();
    let mut best = None::<(
        ConstrainedCavity,
        ConstrainedCavityNode,
        Vec<ConstrainedCavityRefillTetrahedron>,
        f64,
        usize,
    )>;
    for face in faces {
        let Some((split_cavity, split_node, split_tetrahedra)) =
            best_boundary_face_edge_split_completion(
                *face,
                cavity,
                boundary_nodes,
                boundary_triangles,
                refill_tetrahedra,
                options,
            )?
        else {
            continue;
        };
        let mut candidate_tetrahedra = refill_tetrahedra.to_vec();
        candidate_tetrahedra.extend(split_tetrahedra.clone());
        let candidate_delta = refill_boundary_face_delta(&split_cavity, &candidate_tetrahedra)?;
        let candidate_delta_count =
            candidate_delta.missing.len() + candidate_delta.unexpected.len();
        if candidate_delta_count >= current_delta_count {
            continue;
        }
        let min_quality = split_tetrahedra
            .iter()
            .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min);
        if best
            .as_ref()
            .is_none_or(|(_, _, _, best_quality, best_delta_count)| {
                candidate_delta_count < *best_delta_count
                    || (candidate_delta_count == *best_delta_count && min_quality > *best_quality)
            })
        {
            best = Some((
                split_cavity,
                split_node,
                split_tetrahedra,
                min_quality,
                candidate_delta_count,
            ));
        }
    }
    Ok(
        best.map(|(split_cavity, split_node, split_tetrahedra, _, _)| {
            (split_cavity, split_node, split_tetrahedra)
        }),
    )
}

pub(super) fn best_boundary_face_three_edge_split_completion_for_faces(
    faces: &[[u32; 3]],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    refill_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    options: ConstrainedCavityRefillOptions,
) -> Result<
    Option<(
        ConstrainedCavity,
        Vec<ConstrainedCavityNode>,
        Vec<ConstrainedCavityRefillTetrahedron>,
    )>,
    ConstrainedCavityValidationError,
> {
    let current_delta = refill_boundary_face_delta(cavity, refill_tetrahedra)?;
    let current_delta_count = current_delta.missing.len() + current_delta.unexpected.len();
    let mut best = None::<(
        ConstrainedCavity,
        Vec<ConstrainedCavityNode>,
        Vec<ConstrainedCavityRefillTetrahedron>,
        f64,
        usize,
    )>;
    for face in faces {
        let Some((split_cavity, split_nodes, split_tetrahedra)) =
            best_boundary_face_three_edge_split_completion(
                *face,
                cavity,
                boundary_nodes,
                boundary_triangles,
                refill_tetrahedra,
                options,
            )?
        else {
            continue;
        };
        let mut candidate_tetrahedra = refill_tetrahedra.to_vec();
        candidate_tetrahedra.extend(split_tetrahedra.clone());
        let candidate_delta = refill_boundary_face_delta(&split_cavity, &candidate_tetrahedra)?;
        let candidate_delta_count =
            candidate_delta.missing.len() + candidate_delta.unexpected.len();
        if candidate_delta_count >= current_delta_count {
            continue;
        }
        let min_quality = split_tetrahedra
            .iter()
            .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min);
        if best
            .as_ref()
            .is_none_or(|(_, _, _, best_quality, best_delta_count)| {
                candidate_delta_count < *best_delta_count
                    || (candidate_delta_count == *best_delta_count && min_quality > *best_quality)
            })
        {
            best = Some((
                split_cavity,
                split_nodes,
                split_tetrahedra,
                min_quality,
                candidate_delta_count,
            ));
        }
    }
    Ok(
        best.map(|(split_cavity, split_nodes, split_tetrahedra, _, _)| {
            (split_cavity, split_nodes, split_tetrahedra)
        }),
    )
}

pub(super) fn best_boundary_face_edge_split_completion(
    face: [u32; 3],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    refill_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    options: ConstrainedCavityRefillOptions,
) -> Result<
    Option<(
        ConstrainedCavity,
        ConstrainedCavityNode,
        Vec<ConstrainedCavityRefillTetrahedron>,
    )>,
    ConstrainedCavityValidationError,
> {
    let split_candidates = boundary_face_edge_split_node_candidates(face, boundary_nodes);
    let mut best = None::<(
        [u32; 2],
        ConstrainedCavityNode,
        Vec<ConstrainedCavityRefillTetrahedron>,
        f64,
    )>;
    for cap_node_id in cavity_boundary_node_ids(cavity) {
        if face.contains(&cap_node_id) {
            continue;
        }
        for (edge, split_node) in &split_candidates {
            let Some(child_tetrahedra) = edge_split_completion_tetrahedra_for_node(
                face,
                *edge,
                cap_node_id,
                split_node,
                boundary_nodes,
                options,
            ) else {
                continue;
            };
            if child_tetrahedra.iter().any(|tetrahedron| {
                let tetrahedron_points = tetrahedron.node_ids.map(|node_id| {
                    if node_id == split_node.node_id {
                        split_node.coordinates_m
                    } else {
                        boundary_nodes[&node_id]
                    }
                });
                point_in_closed_triangle_surface(
                    tetrahedron_centroid(tetrahedron_points),
                    boundary_triangles,
                    MeshingTolerance::default(),
                ) != PointInClosedSurface::Inside
            }) {
                continue;
            }
            if child_tetrahedra.iter().any(|tetrahedron| {
                refill_tetrahedra.iter().any(|existing| {
                    sorted_tetrahedron_nodes(existing.node_ids)
                        == sorted_tetrahedron_nodes(tetrahedron.node_ids)
                })
            }) {
                continue;
            }
            let min_quality = child_tetrahedra
                .iter()
                .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
                .fold(f64::INFINITY, f64::min);
            if best
                .as_ref()
                .is_none_or(|(_, _, _, best_quality)| min_quality > *best_quality)
            {
                best = Some((*edge, split_node.clone(), child_tetrahedra, min_quality));
            }
        }
    }
    let Some((edge, split_node, split_tetrahedra, _)) = best else {
        return Ok(None);
    };
    let split_faces = split_constrained_cavity_boundary_faces_on_edge(
        &cavity.boundary_faces,
        face,
        edge,
        split_node.node_id,
    )
    .map_err(|_| ConstrainedCavityValidationError::MissingBoundaryFace {
        node_ids: sorted_face(face),
    })?;
    let mut split_cavity = cavity.clone();
    split_cavity.boundary_faces = split_faces;
    validate_constrained_cavity(&split_cavity)?;
    Ok(Some((split_cavity, split_node, split_tetrahedra)))
}

pub(super) fn best_boundary_face_three_edge_split_completion(
    face: [u32; 3],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    refill_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    options: ConstrainedCavityRefillOptions,
) -> Result<
    Option<(
        ConstrainedCavity,
        Vec<ConstrainedCavityNode>,
        Vec<ConstrainedCavityRefillTetrahedron>,
    )>,
    ConstrainedCavityValidationError,
> {
    let split_nodes = boundary_face_mid_edge_split_nodes(face, boundary_nodes);
    let split_node_by_edge = face_edges(face)
        .into_iter()
        .zip(split_nodes.iter())
        .map(|(edge, node)| (sorted_edge(edge), node.node_id))
        .collect::<BTreeMap<_, _>>();
    let split_node_coordinates = split_nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut best = None::<(Vec<ConstrainedCavityRefillTetrahedron>, f64)>;
    for cap_node_id in cavity_boundary_node_ids(cavity) {
        if face.contains(&cap_node_id) {
            continue;
        }
        let Some(child_tetrahedra) = three_edge_split_completion_tetrahedra_for_node(
            face,
            cap_node_id,
            &split_node_by_edge,
            &split_node_coordinates,
            boundary_nodes,
            options,
        ) else {
            continue;
        };
        if child_tetrahedra.iter().any(|tetrahedron| {
            let tetrahedron_points = tetrahedron.node_ids.map(|node_id| {
                split_node_coordinates
                    .get(&node_id)
                    .copied()
                    .unwrap_or_else(|| boundary_nodes[&node_id])
            });
            point_in_closed_triangle_surface(
                tetrahedron_centroid(tetrahedron_points),
                boundary_triangles,
                MeshingTolerance::default(),
            ) != PointInClosedSurface::Inside
        }) {
            continue;
        }
        if child_tetrahedra.iter().any(|tetrahedron| {
            refill_tetrahedra.iter().any(|existing| {
                sorted_tetrahedron_nodes(existing.node_ids)
                    == sorted_tetrahedron_nodes(tetrahedron.node_ids)
            })
        }) {
            continue;
        }
        let min_quality = child_tetrahedra
            .iter()
            .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min);
        if best
            .as_ref()
            .is_none_or(|(_, best_quality)| min_quality > *best_quality)
        {
            best = Some((child_tetrahedra, min_quality));
        }
    }
    let Some((split_tetrahedra, _)) = best else {
        return Ok(None);
    };
    let split_faces = split_constrained_cavity_boundary_faces_on_three_edges(
        &cavity.boundary_faces,
        face,
        split_node_by_edge,
    )
    .map_err(|_| ConstrainedCavityValidationError::MissingBoundaryFace {
        node_ids: sorted_face(face),
    })?;
    let mut split_cavity = cavity.clone();
    split_cavity.boundary_faces = split_faces;
    validate_constrained_cavity(&split_cavity)?;
    Ok(Some((split_cavity, split_nodes, split_tetrahedra)))
}

pub(super) fn best_boundary_face_split_completion(
    face: [u32; 3],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    refill_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    options: ConstrainedCavityRefillOptions,
) -> Result<
    Option<(
        ConstrainedCavity,
        ConstrainedCavityNode,
        Vec<ConstrainedCavityRefillTetrahedron>,
    )>,
    ConstrainedCavityValidationError,
> {
    let split_candidates = boundary_face_split_node_candidates(face, boundary_nodes);
    let mut best = None::<(
        ConstrainedCavityNode,
        Vec<ConstrainedCavityRefillTetrahedron>,
        f64,
    )>;
    for cap_node_id in cavity_boundary_node_ids(cavity) {
        if face.contains(&cap_node_id) {
            continue;
        }
        for split_node in &split_candidates {
            let Some(child_tetrahedra) = split_completion_tetrahedra_for_node(
                face,
                cap_node_id,
                split_node,
                boundary_nodes,
                options,
            ) else {
                continue;
            };
            if child_tetrahedra.iter().any(|tetrahedron| {
                let tetrahedron_points = tetrahedron.node_ids.map(|node_id| {
                    if node_id == split_node.node_id {
                        split_node.coordinates_m
                    } else {
                        boundary_nodes[&node_id]
                    }
                });
                point_in_closed_triangle_surface(
                    tetrahedron_centroid(tetrahedron_points),
                    boundary_triangles,
                    MeshingTolerance::default(),
                ) != PointInClosedSurface::Inside
            }) {
                continue;
            }
            if child_tetrahedra.iter().any(|tetrahedron| {
                refill_tetrahedra.iter().any(|existing| {
                    sorted_tetrahedron_nodes(existing.node_ids)
                        == sorted_tetrahedron_nodes(tetrahedron.node_ids)
                })
            }) {
                continue;
            }
            let min_quality = child_tetrahedra
                .iter()
                .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
                .fold(f64::INFINITY, f64::min);
            if best
                .as_ref()
                .is_none_or(|(_, _, best_quality)| min_quality > *best_quality)
            {
                best = Some((split_node.clone(), child_tetrahedra, min_quality));
            }
        }
    }
    let Some((split_node, split_tetrahedra, _)) = best else {
        return Ok(None);
    };
    let split_faces =
        split_constrained_cavity_boundary_faces(&cavity.boundary_faces, face, split_node.node_id)
            .map_err(|_| ConstrainedCavityValidationError::MissingBoundaryFace {
            node_ids: sorted_face(face),
        })?;
    let mut split_cavity = cavity.clone();
    split_cavity.boundary_faces = split_faces;
    validate_constrained_cavity(&split_cavity)?;
    Ok(Some((split_cavity, split_node, split_tetrahedra)))
}

pub(super) fn best_boundary_face_completion_tetrahedron(
    face: [u32; 3],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    refill_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> Option<ConstrainedCavityRefillTetrahedron> {
    cavity_boundary_node_ids(cavity)
        .into_iter()
        .filter(|node_id| !face.contains(node_id))
        .filter_map(|node_id| {
            let node_ids = [face[0], face[1], face[2], node_id];
            let points = node_ids.map(|id| boundary_nodes[&id]);
            if point_in_closed_triangle_surface(
                tetrahedron_centroid(points),
                boundary_triangles,
                MeshingTolerance::default(),
            ) != PointInClosedSurface::Inside
            {
                return None;
            }
            let tetrahedron =
                raw_refill_tetrahedron_with_rejection_reason(node_ids, points, options).ok()?;
            if refill_tetrahedra.iter().any(|existing| {
                sorted_tetrahedron_nodes(existing.node_ids)
                    == sorted_tetrahedron_nodes(tetrahedron.node_ids)
            }) {
                return None;
            }
            Some(tetrahedron)
        })
        .max_by(|left, right| {
            left.exact_scaled_jacobian
                .total_cmp(&right.exact_scaled_jacobian)
                .then_with(|| right.aspect_ratio.total_cmp(&left.aspect_ratio))
        })
}
