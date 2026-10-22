use super::*;
/// Batch admission shares immutable face-local search inventories. The public
/// single-candidate API retains its direct walk and does not invent a memory grant.
pub(in crate::refinement::insertion) fn validate_candidates(
    pslg: &ExactFacePslg,
    trimmed: &ExactFaceTrimmedDelaunay,
    candidates: &[ExactFaceRefinementCandidate],
    cancellation: &dyn MeshingCancellationSignal,
    control: Option<&dyn runmat_geometry_core::GeometryEvaluationControl>,
) -> Result<(), ExactFaceRefinementError> {
    use crate::refinement::spatial;
    use runmat_geometry_core::GeometryAllocationScope;
    use runmat_meshing_core::quality::spatial_index::SpatialEntry;
    let Some(control) = control else {
        for candidate in candidates {
            validate_candidate(pslg, trimmed, candidate, cancellation)?;
        }
        return Ok(());
    };
    let storage = GeometryAllocationScope::new(control);
    let nodes = spatial::build(
        pslg.vertices.len(),
        pslg.vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| SpatialEntry {
                bounds: spatial::point(vertex.uv),
                payload: index,
            }),
        Some(&storage),
        &pslg.source_face_id,
    )?;
    let segments = spatial::build(
        pslg.segments.len(),
        pslg.segments
            .iter()
            .enumerate()
            .map(|(index, segment)| SpatialEntry {
                bounds: spatial::bounds(
                    segment
                        .vertex_indices
                        .map(|vertex| pslg.vertices[vertex as usize].uv),
                ),
                payload: index,
            }),
        Some(&storage),
        &pslg.source_face_id,
    )?;
    let triangles = spatial::build(
        trimmed.triangles.len(),
        trimmed
            .triangles
            .iter()
            .enumerate()
            .map(|(index, triangle)| SpatialEntry {
                bounds: spatial::bounds(
                    triangle
                        .vertex_indices
                        .map(|vertex| pslg.vertices[vertex as usize].uv),
                ),
                payload: index,
            }),
        Some(&storage),
        &pslg.source_face_id,
    )?;
    for candidate in candidates {
        checkpoint(pslg, cancellation)?;
        if candidate.source_face_id != pslg.source_face_id
            || trimmed.triangles.get(candidate.triangle_index as usize) != Some(&candidate.triangle)
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
        let query = spatial::point(candidate.uv);
        let mut duplicate = false;
        nodes.visit_bounds_with_checkpoint(
            query,
            &mut |entry| {
                duplicate |= pslg.vertices[entry.payload].uv == candidate.uv;
            },
            &mut || checkpoint(pslg, cancellation),
        )?;
        if duplicate {
            return Err(invalid(
                pslg,
                "refinement candidate duplicates an existing vertex",
            ));
        }
        let mut failure = None;
        segments.visit_bounds_with_checkpoint(query, &mut |entry| {
            let endpoints = pslg.segments[entry.payload].vertex_indices.map(|index| pslg.vertices[index as usize].uv);
            match orient2d([endpoints[0], endpoints[1], candidate.uv]) {
                Ok(PredicateSign::Zero) if within_segment_bounds(candidate.uv, endpoints) => failure = Some(invalid(pslg, &format!("{:?} refinement candidate {:?} lies on protected segment {:?} ({:?}) and requires a curve split; source triangle {:?}", candidate.reason, candidate.uv, endpoints, pslg.segments[entry.payload].source, candidate.triangle.vertex_indices.map(|index| (index, pslg.vertices[index as usize].uv, pslg.vertices[index as usize].node_id))))),
                Err(_) => failure = Some(invalid(pslg, "refinement candidate has invalid predicate input")),
                _ => {}
            }
        }, &mut || checkpoint(pslg, cancellation))?;
        if let Some(failure) = failure {
            return Err(failure);
        }
        let mut inside = false;
        triangles.visit_bounds_with_checkpoint(
            query,
            &mut |entry| {
                let vertices = trimmed.triangles[entry.payload]
                    .vertex_indices
                    .map(|index| pslg.vertices[index as usize].uv);
                let mut within = true;
                for [start, finish] in [[0, 1], [1, 2], [2, 0]] {
                    match orient2d([vertices[start], vertices[finish], candidate.uv]) {
                        Ok(sign) => within &= sign != PredicateSign::Negative,
                        Err(_) => {
                            failure = Some(invalid(
                                pslg,
                                "refinement candidate has invalid predicate input",
                            ));
                            within = false;
                        }
                    }
                }
                inside |= within;
            },
            &mut || checkpoint(pslg, cancellation),
        )?;
        if let Some(failure) = failure {
            return Err(failure);
        }
        if !inside {
            return Err(invalid(
                pslg,
                "refinement candidate is outside the current trimmed face domain",
            ));
        }
    }
    Ok(())
}
