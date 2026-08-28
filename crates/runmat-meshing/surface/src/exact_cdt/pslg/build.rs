use std::collections::BTreeMap;

use runmat_geometry_core::{PersistentEntityId, TopologicalOrientation};
use runmat_meshing_core::StableDigest;

use crate::ExactFaceBoundary;

use super::{
    validate_exact_face_pslg, ExactFacePslg, ExactFacePslgError, ExactFacePslgErrorKind,
    ExactFacePslgLoop, ExactFacePslgLoopSource, ExactFacePslgSegment, ExactFacePslgSegmentSource,
    ExactFacePslgVertex, MAX_FACE_PSLG_ITEMS,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VertexKey {
    node_id: StableDigest,
    seam_image: Option<u8>,
    uv_bits: [u64; 2],
}

pub(crate) struct PslgSegmentInput {
    pub source: ExactFacePslgSegmentSource,
    pub endpoints: [ExactFacePslgVertex; 2],
    pub edge_parameters: Option<[f64; 2]>,
}

pub(crate) struct PslgLoopInput {
    pub source: ExactFacePslgLoopSource,
    pub orientation: TopologicalOrientation,
    pub segments: Vec<PslgSegmentInput>,
}

pub fn build_exact_face_pslg(
    boundary: &ExactFaceBoundary,
) -> Result<ExactFacePslg, ExactFacePslgError> {
    let loops = std::iter::once(&boundary.outer_loop)
        .chain(&boundary.inner_loops)
        .map(|loop_boundary| PslgLoopInput {
            source: ExactFacePslgLoopSource::ExactWire {
                source_wire_id: loop_boundary.source_wire_id.clone(),
            },
            orientation: loop_boundary.orientation,
            segments: loop_boundary
                .segments
                .iter()
                .map(|segment| PslgSegmentInput {
                    source: ExactFacePslgSegmentSource::ExactTrim {
                        source_coedge_id: segment.source_coedge_id.clone(),
                        source_edge_id: segment.source_edge_id.clone(),
                    },
                    endpoints: [0, 1].map(|endpoint| ExactFacePslgVertex {
                        node_id: segment.node_ids[endpoint],
                        seam_image: segment.seam_image,
                        uv: segment.node_uv[endpoint],
                    }),
                    edge_parameters: Some(segment.edge_parameters),
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let pslg = build_canonical_pslg(&boundary.source_face_id, loops)?;
    validate_exact_face_pslg(&pslg, boundary)?;
    Ok(pslg)
}

pub(crate) fn build_canonical_pslg(
    source_face_id: &PersistentEntityId,
    loops: Vec<PslgLoopInput>,
) -> Result<ExactFacePslg, ExactFacePslgError> {
    let segment_count = loops
        .iter()
        .map(|loop_input| loop_input.segments.len())
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(|| resource_error(source_face_id, "face PSLG segment count overflow"))?;
    if loops.is_empty() || segment_count == 0 || segment_count > MAX_FACE_PSLG_ITEMS {
        return Err(resource_error(
            source_face_id,
            "face PSLG loop or segment inventory is empty or exceeds its hard bound",
        ));
    }

    let mut keys = loops
        .iter()
        .flat_map(|loop_input| &loop_input.segments)
        .flat_map(|segment| segment.endpoints.map(vertex_key))
        .collect::<Vec<_>>();
    keys.sort_by(compare_keys);
    keys.dedup();
    if keys.len() > MAX_FACE_PSLG_ITEMS || keys.len() > u32::MAX as usize {
        return Err(resource_error(
            source_face_id,
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
    let mut pslg_loops = Vec::with_capacity(loops.len());
    for loop_input in loops {
        let first_segment = u32::try_from(segments.len())
            .map_err(|_| resource_error(source_face_id, "face PSLG segment offset exceeds u32"))?;
        let loop_segment_count = u32::try_from(loop_input.segments.len())
            .map_err(|_| resource_error(source_face_id, "face PSLG loop length exceeds u32"))?;
        for segment in loop_input.segments {
            segments.push(ExactFacePslgSegment {
                source: segment.source,
                vertex_indices: segment
                    .endpoints
                    .map(|endpoint| vertex_index[&vertex_key(endpoint)]),
                edge_parameters: segment.edge_parameters,
            });
        }
        pslg_loops.push(ExactFacePslgLoop {
            source: loop_input.source,
            orientation: loop_input.orientation,
            first_segment,
            segment_count: loop_segment_count,
        });
    }
    Ok(ExactFacePslg {
        source_face_id: source_face_id.clone(),
        vertices,
        segments,
        loops: pslg_loops,
    })
}

fn vertex_key(vertex: ExactFacePslgVertex) -> VertexKey {
    VertexKey {
        node_id: vertex.node_id,
        seam_image: vertex.seam_image,
        uv_bits: vertex.uv.map(f64::to_bits),
    }
}

fn compare_keys(left: &VertexKey, right: &VertexKey) -> std::cmp::Ordering {
    left.node_id
        .cmp(&right.node_id)
        .then_with(|| left.seam_image.cmp(&right.seam_image))
        .then_with(|| f64::from_bits(left.uv_bits[0]).total_cmp(&f64::from_bits(right.uv_bits[0])))
        .then_with(|| f64::from_bits(left.uv_bits[1]).total_cmp(&f64::from_bits(right.uv_bits[1])))
}

fn resource_error(source_face_id: &PersistentEntityId, reason: &str) -> ExactFacePslgError {
    ExactFacePslgError::new(
        ExactFacePslgErrorKind::ResourceLimit,
        source_face_id,
        reason,
    )
}
