use super::*;

pub(in super::super::super) fn best_boundary_face_three_edge_split_completion_for_faces(
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
