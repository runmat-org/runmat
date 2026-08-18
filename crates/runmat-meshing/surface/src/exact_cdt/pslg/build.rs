use std::collections::BTreeMap;

use runmat_meshing_core::StableDigest;

use crate::{ExactFaceBoundary, ExactFaceBoundaryLoop};

use super::{
    validate_exact_face_pslg, ExactFacePslg, ExactFacePslgError, ExactFacePslgErrorKind,
    ExactFacePslgLoop, ExactFacePslgSegment, ExactFacePslgVertex,
};

const MAX_FACE_PSLG_ITEMS: usize = 10_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VertexKey {
    node_id: StableDigest,
    seam_image: Option<u8>,
    uv_bits: [u64; 2],
}

pub fn build_exact_face_pslg(
    boundary: &ExactFaceBoundary,
) -> Result<ExactFacePslg, ExactFacePslgError> {
    let segment_count = std::iter::once(&boundary.outer_loop)
        .chain(&boundary.inner_loops)
        .map(|loop_boundary| loop_boundary.segments.len())
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(|| resource_error(boundary, "face PSLG segment count overflow"))?;
    if segment_count == 0 || segment_count > MAX_FACE_PSLG_ITEMS {
        return Err(resource_error(
            boundary,
            "face PSLG segment count is empty or exceeds its hard bound",
        ));
    }

    let mut keys = std::iter::once(&boundary.outer_loop)
        .chain(&boundary.inner_loops)
        .flat_map(|loop_boundary| &loop_boundary.segments)
        .flat_map(|segment| {
            (0..2).map(|endpoint| VertexKey {
                node_id: segment.node_ids[endpoint],
                seam_image: segment.seam_image,
                uv_bits: segment.node_uv[endpoint].map(f64::to_bits),
            })
        })
        .collect::<Vec<_>>();
    keys.sort_by(|left, right| {
        left.node_id
            .cmp(&right.node_id)
            .then_with(|| left.seam_image.cmp(&right.seam_image))
            .then_with(|| {
                f64::from_bits(left.uv_bits[0]).total_cmp(&f64::from_bits(right.uv_bits[0]))
            })
            .then_with(|| {
                f64::from_bits(left.uv_bits[1]).total_cmp(&f64::from_bits(right.uv_bits[1]))
            })
    });
    keys.dedup();
    if keys.len() > MAX_FACE_PSLG_ITEMS || keys.len() > u32::MAX as usize {
        return Err(resource_error(
            boundary,
            "face PSLG vertex count exceeds its hard bound",
        ));
    }
    let vertex_index = keys
        .iter()
        .enumerate()
        .map(|(index, key)| (*key, index as u32))
        .collect::<BTreeMap<_, _>>();
    let vertices = keys
        .iter()
        .map(|key| ExactFacePslgVertex {
            node_id: key.node_id,
            seam_image: key.seam_image,
            uv: key.uv_bits.map(f64::from_bits),
        })
        .collect();

    let mut segments = Vec::with_capacity(segment_count);
    let mut loops = Vec::with_capacity(1 + boundary.inner_loops.len());
    for loop_boundary in std::iter::once(&boundary.outer_loop).chain(&boundary.inner_loops) {
        push_loop(
            &boundary.source_face_id,
            loop_boundary,
            &vertex_index,
            &mut segments,
            &mut loops,
        )?;
    }
    let pslg = ExactFacePslg {
        source_face_id: boundary.source_face_id.clone(),
        vertices,
        segments,
        loops,
    };
    validate_exact_face_pslg(&pslg, boundary)?;
    Ok(pslg)
}

fn push_loop(
    source_face_id: &runmat_geometry_core::PersistentEntityId,
    boundary: &ExactFaceBoundaryLoop,
    vertex_index: &BTreeMap<VertexKey, u32>,
    segments: &mut Vec<ExactFacePslgSegment>,
    loops: &mut Vec<ExactFacePslgLoop>,
) -> Result<(), ExactFacePslgError> {
    let first_segment = u32::try_from(segments.len()).map_err(|_| {
        ExactFacePslgError::new(
            ExactFacePslgErrorKind::ResourceLimit,
            source_face_id,
            "face PSLG segment offset exceeds u32",
        )
    })?;
    for segment in &boundary.segments {
        let key = |endpoint: usize| VertexKey {
            node_id: segment.node_ids[endpoint],
            seam_image: segment.seam_image,
            uv_bits: segment.node_uv[endpoint].map(f64::to_bits),
        };
        segments.push(ExactFacePslgSegment {
            source_coedge_id: segment.source_coedge_id.clone(),
            source_edge_id: segment.source_edge_id.clone(),
            vertex_indices: [vertex_index[&key(0)], vertex_index[&key(1)]],
            edge_parameters: segment.edge_parameters,
        });
    }
    loops.push(ExactFacePslgLoop {
        source_wire_id: boundary.source_wire_id.clone(),
        orientation: boundary.orientation,
        first_segment,
        segment_count: boundary.segments.len() as u32,
    });
    Ok(())
}

fn resource_error(boundary: &ExactFaceBoundary, reason: &str) -> ExactFacePslgError {
    ExactFacePslgError::new(
        ExactFacePslgErrorKind::ResourceLimit,
        &boundary.source_face_id,
        reason,
    )
}
