use std::collections::BTreeSet;

use crate::ExactFaceBoundary;

use super::{
    exact_face_interior_node_id, ExactFacePslg, ExactFacePslgError, ExactFacePslgErrorKind,
    ExactFacePslgLoopSource, ExactFacePslgSegmentSource, MAX_FACE_PSLG_ITEMS,
};

pub fn validate_exact_face_pslg(
    pslg: &ExactFacePslg,
    boundary: &ExactFaceBoundary,
) -> Result<(), ExactFacePslgError> {
    if pslg.vertices.len() > MAX_FACE_PSLG_ITEMS
        || pslg.segments.len() > MAX_FACE_PSLG_ITEMS
        || pslg.loops.len() > MAX_FACE_PSLG_ITEMS
    {
        return Err(ExactFacePslgError::new(
            ExactFacePslgErrorKind::ResourceLimit,
            &boundary.source_face_id,
            "face PSLG inventory exceeds its hard bound",
        ));
    }
    if pslg.source_face_id != boundary.source_face_id
        || pslg.loops.len() != 1 + boundary.inner_loops.len()
        || pslg.vertices.is_empty()
        || pslg.segments.is_empty()
    {
        return Err(invalid(boundary, "face PSLG inventory is inconsistent"));
    }
    let unique_vertices = pslg
        .vertices
        .iter()
        .map(|vertex| {
            (
                vertex.node_id,
                vertex.seam_image,
                vertex.uv.map(f64::to_bits),
            )
        })
        .collect::<BTreeSet<_>>();
    if unique_vertices.len() != pslg.vertices.len()
        || pslg
            .vertices
            .iter()
            .flat_map(|vertex| vertex.uv)
            .any(|value| !value.is_finite())
    {
        return Err(invalid(boundary, "face PSLG vertices are not canonical"));
    }
    if pslg.vertices.windows(2).any(|pair| {
        pair[0]
            .node_id
            .cmp(&pair[1].node_id)
            .then_with(|| pair[0].seam_image.cmp(&pair[1].seam_image))
            .then_with(|| pair[0].uv[0].total_cmp(&pair[1].uv[0]))
            .then_with(|| pair[0].uv[1].total_cmp(&pair[1].uv[1]))
            .is_gt()
    }) {
        return Err(invalid(boundary, "face PSLG vertex order is not canonical"));
    }

    let boundary_loops = std::iter::once(&boundary.outer_loop).chain(&boundary.inner_loops);
    let mut expected_offset = 0usize;
    let mut referenced_vertices = BTreeSet::new();
    for (actual_loop, boundary_loop) in pslg.loops.iter().zip(boundary_loops) {
        let expected_loop_source = ExactFacePslgLoopSource::ExactWire {
            source_wire_id: boundary_loop.source_wire_id.clone(),
        };
        if actual_loop.source != expected_loop_source
            || actual_loop.orientation != boundary_loop.orientation
            || actual_loop.first_segment as usize != expected_offset
            || actual_loop.segment_count as usize != boundary_loop.segments.len()
            || actual_loop.segment_count == 0
        {
            return Err(invalid(boundary, "face PSLG loop range is inconsistent"));
        }
        let end = expected_offset
            .checked_add(boundary_loop.segments.len())
            .ok_or_else(|| invalid(boundary, "face PSLG loop range overflow"))?;
        let actual_segments = pslg
            .segments
            .get(expected_offset..end)
            .ok_or_else(|| invalid(boundary, "face PSLG loop range is incomplete"))?;
        for (actual, expected) in actual_segments.iter().zip(&boundary_loop.segments) {
            referenced_vertices.extend(actual.vertex_indices);
            let Some(start) = pslg.vertices.get(actual.vertex_indices[0] as usize) else {
                return Err(invalid(boundary, "face PSLG segment start is absent"));
            };
            let Some(finish) = pslg.vertices.get(actual.vertex_indices[1] as usize) else {
                return Err(invalid(boundary, "face PSLG segment end is absent"));
            };
            let expected_source = ExactFacePslgSegmentSource::ExactTrim {
                source_coedge_id: expected.source_coedge_id.clone(),
                source_edge_id: expected.source_edge_id.clone(),
            };
            if actual.source != expected_source
                || start.node_id != expected.node_ids[0]
                || finish.node_id != expected.node_ids[1]
                || actual.edge_parameters != Some(expected.edge_parameters)
                || start.seam_image != expected.seam_image
                || finish.seam_image != expected.seam_image
                || start.uv != expected.node_uv[0]
                || finish.uv != expected.node_uv[1]
                || actual.vertex_indices[0] == actual.vertex_indices[1]
                || actual
                    .edge_parameters
                    .is_none_or(|parameters| parameters.iter().any(|value| !value.is_finite()))
            {
                return Err(invalid(boundary, "face PSLG segment differs from boundary"));
            }
        }
        if actual_segments
            .iter()
            .zip(actual_segments.iter().cycle().skip(1))
            .take(actual_segments.len())
            .any(|(left, right)| left.vertex_indices[1] != right.vertex_indices[0])
        {
            return Err(ExactFacePslgError::new(
                ExactFacePslgErrorKind::InvalidTopology,
                &boundary.source_face_id,
                "face boundary requires another parametric chart",
            ));
        }
        expected_offset = end;
    }
    if expected_offset != pslg.segments.len() {
        return Err(invalid(boundary, "face PSLG contains unowned segments"));
    }
    for (index, vertex) in pslg.vertices.iter().enumerate() {
        if !referenced_vertices.contains(&(index as u32))
            && (vertex.seam_image.is_some()
                || vertex.node_id != exact_face_interior_node_id(&pslg.source_face_id, vertex.uv))
        {
            return Err(invalid(
                boundary,
                "face PSLG interior vertex identity is not canonical",
            ));
        }
    }
    Ok(())
}

fn invalid(boundary: &ExactFaceBoundary, reason: &str) -> ExactFacePslgError {
    ExactFacePslgError::new(
        ExactFacePslgErrorKind::InvalidBoundary,
        &boundary.source_face_id,
        reason,
    )
}
