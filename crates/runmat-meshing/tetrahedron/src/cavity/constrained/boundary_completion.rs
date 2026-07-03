use std::collections::BTreeMap;

use runmat_meshing_core::predicate::{Point3, Triangle3};

mod candidates;
mod selection;

pub(super) use candidates::{
    best_boundary_face_completion_tetrahedron, best_boundary_face_edge_split_completion,
    best_boundary_face_split_completion, best_boundary_face_three_edge_split_completion,
};
pub(super) use selection::{
    best_boundary_face_completion_tetrahedron_for_faces,
    best_boundary_face_edge_split_completion_for_faces,
    best_boundary_face_split_completion_for_faces,
    best_boundary_face_three_edge_split_completion_for_faces,
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
