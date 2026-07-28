use super::*;

pub(in super::super::super) fn best_boundary_face_completion_tetrahedron_for_faces(
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
