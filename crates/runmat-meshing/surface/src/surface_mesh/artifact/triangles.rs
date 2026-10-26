//! One admitted reference buffer serves identity and facet checks; node lookup
//! borrows the independently checked canonical node array.
use super::*;
use crate::face_mesh::{canonical_triangle, exact_face_triangle_id, valid_triangle_measures};
use crate::ExactFaceMeshTriangle;

fn facet(triangle: &ExactFaceMeshTriangle) -> [StableDigest; 3] {
    let mut ids = triangle.node_ids;
    ids.sort_unstable();
    ids
}

pub(super) fn validate(
    mesh: &ExactSurfaceMesh,
    control: Option<&dyn MeshingCancellationSignal>,
    allocations: Option<&dyn runmat_geometry_core::GeometryEvaluationControl>,
) -> Result<(), ExactSurfaceMeshError> {
    let _triangle_storage = super::super::join::nodes::reserve::<&ExactFaceMeshTriangle>(
        mesh.triangles.len(),
        allocations,
    )?;
    let _coverage_storage =
        super::super::join::nodes::reserve::<bool>(mesh.face_ids.len(), allocations)?;
    let mut references = Vec::with_capacity(mesh.triangles.len());
    let mut covered = vec![false; mesh.face_ids.len()];
    let mut maximum_chordal = 0.0_f64;
    let mut maximum_normal = 0.0_f64;
    for triangle in &mesh.triangles {
        checkpoint(control)?;
        let owner = mesh.face_ids.binary_search(&triangle.source_face_id);
        let sorted = facet(triangle);
        if owner.is_err()
            || triangle.chart_id == StableDigest::ZERO
            || triangle.triangle_id != exact_face_triangle_id(triangle.chart_id, triangle.node_ids)
            || canonical_triangle(triangle.node_ids).0 != triangle.node_ids
            || sorted[0] == sorted[1]
            || sorted[1] == sorted[2]
            || triangle.node_ids.iter().any(|id| {
                mesh.nodes
                    .binary_search_by_key(id, |node| node.node_id)
                    .ok()
                    .map(|index| &mesh.nodes[index])
                    .is_none_or(|node| {
                        !node.uses.iter().any(|usage| {
                            usage.source_face_id == triangle.source_face_id
                                && usage.chart_id == triangle.chart_id
                        })
                    })
            })
            || triangle.acceptance_sample_count == 0
            || !valid_triangle_measures(triangle)
        {
            return Err(invalid("published surface triangle is invalid"));
        }
        covered[owner.expect("triangle owner checked")] = true;
        references.push(triangle);
        maximum_chordal = maximum_chordal.max(triangle.accepted_chordal_deviation_m);
        maximum_normal = maximum_normal.max(triangle.accepted_normal_deviation_rad);
    }
    runmat_cpu_execution::sort_in_place_with_checkpoint(
        &mut references,
        |left, right| left.triangle_id.cmp(&right.triangle_id),
        || checkpoint(control),
    )?;
    if references
        .windows(2)
        .any(|pair| pair[0].triangle_id == pair[1].triangle_id)
    {
        return Err(invalid("published surface triangle is invalid"));
    }
    runmat_cpu_execution::sort_in_place_with_checkpoint(
        &mut references,
        |left, right| {
            left.source_face_id
                .cmp(&right.source_face_id)
                .then_with(|| facet(left).cmp(&facet(right)))
        },
        || checkpoint(control),
    )?;
    if references.windows(2).any(|pair| {
        pair[0].source_face_id == pair[1].source_face_id && facet(pair[0]) == facet(pair[1])
    }) {
        return Err(invalid("published surface triangle is invalid"));
    }
    if covered.iter().any(|covered| !covered)
        || mesh.maximum_chordal_deviation_m.canonical_value() != maximum_chordal
        || mesh.maximum_normal_deviation_rad != maximum_normal
    {
        return Err(invalid(
            "published surface face coverage or quality summary is invalid",
        ));
    }
    Ok(())
}
