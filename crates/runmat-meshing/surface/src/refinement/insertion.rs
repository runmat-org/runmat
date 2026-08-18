use runmat_meshing_core::{predicate::orient2d, MeshingCancellationSignal, PredicateSign};

use crate::exact_cdt::MAX_FACE_PSLG_ITEMS;
use crate::{
    carve_exact_face_domain, exact_face_interior_node_id, recover_exact_face_segments,
    triangulate_exact_face_pslg, validate_exact_face_trimmed_delaunay, ExactFaceBoundary,
    ExactFaceDelaunayError, ExactFaceDelaunayErrorKind, ExactFaceDelaunayOptions, ExactFacePslg,
    ExactFacePslgVertex, ExactFaceTrimmedDelaunay,
};

use super::{
    ExactFaceRefinedTopology, ExactFaceRefinementCandidate, ExactFaceRefinementError,
    ExactFaceRefinementErrorKind,
};

pub fn insert_exact_face_refinement_candidate(
    boundary: &ExactFaceBoundary,
    topology: &ExactFaceRefinedTopology,
    candidate: &ExactFaceRefinementCandidate,
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
) -> Result<ExactFaceRefinedTopology, ExactFaceRefinementError> {
    let pslg = &topology.pslg;
    validate_exact_face_trimmed_delaunay(
        &topology.trimmed,
        &topology.constrained,
        pslg,
        boundary,
        cancellation,
        options,
    )
    .map_err(map_delaunay)?;
    validate_candidate(pslg, &topology.trimmed, candidate)?;

    let refined_pslg = append_interior_vertex(pslg, candidate.uv)?;
    let delaunay = triangulate_exact_face_pslg(&refined_pslg, boundary, cancellation, options)
        .map_err(map_delaunay)?;
    let refined_constrained =
        recover_exact_face_segments(&delaunay, &refined_pslg, boundary, cancellation, options)
            .map_err(map_delaunay)?;
    let refined_trimmed = carve_exact_face_domain(
        &refined_constrained,
        &refined_pslg,
        boundary,
        cancellation,
        options,
    )
    .map_err(map_delaunay)?;
    Ok(ExactFaceRefinedTopology {
        pslg: refined_pslg,
        constrained: refined_constrained,
        trimmed: refined_trimmed,
    })
}

fn validate_candidate(
    pslg: &ExactFacePslg,
    trimmed: &ExactFaceTrimmedDelaunay,
    candidate: &ExactFaceRefinementCandidate,
) -> Result<(), ExactFaceRefinementError> {
    if candidate.source_face_id != pslg.source_face_id
        || candidate.triangle_index as usize >= trimmed.triangles.len()
        || trimmed.triangles[candidate.triangle_index as usize] != candidate.triangle
        || candidate
            .uv
            .iter()
            .any(|coordinate| !coordinate.is_finite())
    {
        return Err(invalid(
            pslg,
            "refinement candidate does not identify current face topology",
        ));
    }
    if pslg.vertices.iter().any(|vertex| vertex.uv == candidate.uv) {
        return Err(invalid(
            pslg,
            "refinement candidate duplicates an existing vertex",
        ));
    }
    for segment in &pslg.segments {
        let endpoints = segment
            .vertex_indices
            .map(|index| pslg.vertices[index as usize].uv);
        let sign = orient2d([endpoints[0], endpoints[1], candidate.uv])
            .map_err(|_| invalid(pslg, "refinement candidate has invalid predicate input"))?;
        if sign == PredicateSign::Zero && within_segment_bounds(candidate.uv, endpoints) {
            return Err(invalid(
                pslg,
                "refinement candidate lies on a protected segment and requires a curve split",
            ));
        }
    }
    let mut inside = false;
    for triangle in &trimmed.triangles {
        let vertices = triangle
            .vertex_indices
            .map(|index| pslg.vertices[index as usize].uv);
        let signs = [
            orient2d([vertices[0], vertices[1], candidate.uv]),
            orient2d([vertices[1], vertices[2], candidate.uv]),
            orient2d([vertices[2], vertices[0], candidate.uv]),
        ];
        let signs = signs
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| invalid(pslg, "refinement candidate has invalid predicate input"))?;
        if signs.iter().all(|sign| *sign != PredicateSign::Negative) {
            inside = true;
            break;
        }
    }
    if !inside {
        return Err(invalid(
            pslg,
            "refinement candidate is outside the current trimmed face domain",
        ));
    }
    Ok(())
}

fn append_interior_vertex(
    pslg: &ExactFacePslg,
    uv: [f64; 2],
) -> Result<ExactFacePslg, ExactFaceRefinementError> {
    if pslg.vertices.len() >= MAX_FACE_PSLG_ITEMS || pslg.vertices.len() >= u32::MAX as usize {
        return Err(delaunay_error(
            pslg,
            ExactFaceDelaunayErrorKind::ResourceLimit,
            "face vertex inventory exceeds its hard bound or index capacity",
        ));
    }
    let old_count = pslg.vertices.len();
    let mut indexed = pslg
        .vertices
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    indexed.push((
        old_count,
        ExactFacePslgVertex {
            node_id: exact_face_interior_node_id(&pslg.source_face_id, uv),
            seam_image: None,
            uv,
        },
    ));
    indexed.sort_by(|(_, left), (_, right)| {
        left.node_id
            .cmp(&right.node_id)
            .then_with(|| left.seam_image.cmp(&right.seam_image))
            .then_with(|| left.uv[0].total_cmp(&right.uv[0]))
            .then_with(|| left.uv[1].total_cmp(&right.uv[1]))
    });
    let mut remap = vec![0u32; indexed.len()];
    let vertices = indexed
        .into_iter()
        .enumerate()
        .map(|(new_index, (old_index, vertex))| {
            remap[old_index] = new_index as u32;
            vertex
        })
        .collect();
    let mut refined = pslg.clone();
    refined.vertices = vertices;
    for segment in &mut refined.segments {
        segment.vertex_indices = segment.vertex_indices.map(|index| remap[index as usize]);
    }
    Ok(refined)
}

fn within_segment_bounds(point: [f64; 2], endpoints: [[f64; 2]; 2]) -> bool {
    (0..2).all(|axis| {
        point[axis] >= endpoints[0][axis].min(endpoints[1][axis])
            && point[axis] <= endpoints[0][axis].max(endpoints[1][axis])
    })
}

fn map_delaunay(error: ExactFaceDelaunayError) -> ExactFaceRefinementError {
    ExactFaceRefinementError::new(
        ExactFaceRefinementErrorKind::Delaunay(error.kind),
        &error.source_face_id,
        error.reason,
    )
}

fn delaunay_error(
    pslg: &ExactFacePslg,
    kind: ExactFaceDelaunayErrorKind,
    reason: &str,
) -> ExactFaceRefinementError {
    ExactFaceRefinementError::new(
        ExactFaceRefinementErrorKind::Delaunay(kind),
        &pslg.source_face_id,
        reason,
    )
}

fn invalid(pslg: &ExactFacePslg, reason: &str) -> ExactFaceRefinementError {
    ExactFaceRefinementError::new(
        ExactFaceRefinementErrorKind::InvalidGeometry,
        &pslg.source_face_id,
        reason,
    )
}
