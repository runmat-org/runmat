use std::collections::BTreeMap;

use crate::predicate::{Point3, Triangle3};

mod candidates;

pub(super) use candidates::{
    best_boundary_face_completion_tetrahedron, best_boundary_face_edge_split_completion,
    best_boundary_face_split_completion, best_boundary_face_three_edge_split_completion,
};

use super::{
    refill_faces::refill_boundary_face_delta, ConstrainedCavity, ConstrainedCavityNode,
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
