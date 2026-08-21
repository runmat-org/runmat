use std::collections::BTreeMap;

use runmat_meshing_core::MeshingCancellationSignal;

use crate::exact_cdt::{
    carve_validated_face_domain, insert_pslg_vertices, recover_validated_face_segments,
    triangulate_validated_face_pslg, validate_face_constrained_topology,
    validate_face_trimmed_topology,
};
use crate::{
    exact_face_chart_cut_node_id, ExactFaceDelaunayError, ExactFaceDelaunayOptions, ExactFacePslg,
    ExactFacePslgSegment, ExactFacePslgSegmentSource, ExactFacePslgVertex,
};

use super::{
    ExactChartCutSplit, ExactFaceRefinedTopology, ExactFaceRefinementError,
    ExactFaceRefinementErrorKind,
};

pub fn split_exact_face_chart_cut(
    current: &ExactFaceRefinedTopology,
    split: &ExactChartCutSplit,
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
) -> Result<ExactFaceRefinedTopology, ExactFaceRefinementError> {
    validate_current(current, cancellation, options)?;
    let pslg = apply_split(&current.pslg, split)?;
    let delaunay =
        triangulate_validated_face_pslg(&pslg, cancellation, options).map_err(map_delaunay)?;
    let constrained = recover_validated_face_segments(&delaunay, &pslg, cancellation, options)
        .map_err(map_delaunay)?;
    let trimmed = carve_validated_face_domain(&constrained, &pslg, cancellation, options)
        .map_err(map_delaunay)?;
    let result = ExactFaceRefinedTopology {
        pslg,
        constrained,
        trimmed,
    };
    validate_exact_face_chart_cut_split_result(&result, current, split, cancellation, options)?;
    Ok(result)
}

pub fn validate_exact_face_chart_cut_split_result(
    result: &ExactFaceRefinedTopology,
    previous: &ExactFaceRefinedTopology,
    split: &ExactChartCutSplit,
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
) -> Result<(), ExactFaceRefinementError> {
    validate_current(previous, cancellation, options)?;
    let expected = apply_split(&previous.pslg, split)?;
    if result.pslg != expected {
        return Err(invalid(
            &previous.pslg,
            "chart-cut split PSLG differs from canonical reconstruction",
        ));
    }
    validate_face_constrained_topology(&result.constrained, &result.pslg, cancellation, options)
        .map_err(map_delaunay)?;
    validate_face_trimmed_topology(
        &result.trimmed,
        &result.constrained,
        &result.pslg,
        cancellation,
        options,
    )
    .map_err(map_delaunay)
}

fn apply_split(
    pslg: &ExactFacePslg,
    split: &ExactChartCutSplit,
) -> Result<ExactFacePslg, ExactFaceRefinementError> {
    validate_split(pslg, split)?;
    let additions = split.images.map(|image| ExactFacePslgVertex {
        node_id: split.node_id,
        seam_image: None,
        uv: image.midpoint_uv,
    });
    let (mut updated, midpoint_indices) =
        insert_pslg_vertices(pslg, &additions).map_err(|reason| invalid(pslg, reason))?;
    let midpoint_by_segment = split
        .images
        .iter()
        .zip(midpoint_indices)
        .map(|(image, midpoint)| (image.pslg_segment_index as usize, midpoint))
        .collect::<BTreeMap<_, _>>();
    let mut segments = Vec::with_capacity(updated.segments.len() + split.images.len());
    for (index, segment) in updated.segments.iter().enumerate() {
        if let Some(midpoint) = midpoint_by_segment.get(&index) {
            segments.push(ExactFacePslgSegment {
                source: segment.source.clone(),
                vertex_indices: [segment.vertex_indices[0], *midpoint],
                edge_parameters: None,
            });
            segments.push(ExactFacePslgSegment {
                source: segment.source.clone(),
                vertex_indices: [*midpoint, segment.vertex_indices[1]],
                edge_parameters: None,
            });
        } else {
            segments.push(segment.clone());
        }
    }
    let mut added_before = 0u32;
    for loop_record in &mut updated.loops {
        let original_start = loop_record.first_segment as usize;
        let original_end = original_start + loop_record.segment_count as usize;
        let added = split
            .images
            .iter()
            .filter(|image| {
                (original_start..original_end).contains(&(image.pslg_segment_index as usize))
            })
            .count() as u32;
        loop_record.first_segment += added_before;
        loop_record.segment_count += added;
        added_before += added;
    }
    updated.segments = segments;
    Ok(updated)
}

fn validate_split(
    pslg: &ExactFacePslg,
    split: &ExactChartCutSplit,
) -> Result<(), ExactFaceRefinementError> {
    if split.source_face_id != pslg.source_face_id
        || split.cut_id == runmat_meshing_core::StableDigest::ZERO
        || split.node_id == runmat_meshing_core::StableDigest::ZERO
        || split.images[0].pslg_segment_index >= split.images[1].pslg_segment_index
    {
        return Err(invalid(pslg, "chart-cut split identity is inconsistent"));
    }
    let mut endpoint_node_ids = None;
    for image in split.images {
        let Some(segment) = pslg.segments.get(image.pslg_segment_index as usize) else {
            return Err(invalid(pslg, "chart-cut split segment is absent"));
        };
        if segment.source
            != (ExactFacePslgSegmentSource::ChartCut {
                cut_id: split.cut_id,
            })
            || segment.edge_parameters.is_some()
            || segment.vertex_indices != image.vertex_indices
        {
            return Err(invalid(pslg, "chart-cut split segment provenance is stale"));
        }
        let endpoints = image
            .vertex_indices
            .map(|index| pslg.vertices[index as usize]);
        let midpoint = [
            endpoints[0].uv[0] * 0.5 + endpoints[1].uv[0] * 0.5,
            endpoints[0].uv[1] * 0.5 + endpoints[1].uv[1] * 0.5,
        ];
        if midpoint != image.midpoint_uv || midpoint.iter().any(|value| !value.is_finite()) {
            return Err(invalid(pslg, "chart-cut split midpoint is inconsistent"));
        }
        let node_ids = endpoints.map(|vertex| vertex.node_id);
        if let Some(first) = endpoint_node_ids.replace(node_ids) {
            if first != [node_ids[1], node_ids[0]] {
                return Err(invalid(
                    pslg,
                    "chart-cut split images do not reverse shared endpoint identities",
                ));
            }
        }
    }
    let endpoint_node_ids = endpoint_node_ids.unwrap();
    if split.node_id != exact_face_chart_cut_node_id(split.cut_id, endpoint_node_ids)
        || pslg
            .vertices
            .iter()
            .any(|vertex| vertex.node_id == split.node_id)
    {
        return Err(invalid(
            pslg,
            "chart-cut split node identity is not canonical",
        ));
    }
    Ok(())
}

fn validate_current(
    current: &ExactFaceRefinedTopology,
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
) -> Result<(), ExactFaceRefinementError> {
    validate_face_constrained_topology(&current.constrained, &current.pslg, cancellation, options)
        .map_err(map_delaunay)?;
    validate_face_trimmed_topology(
        &current.trimmed,
        &current.constrained,
        &current.pslg,
        cancellation,
        options,
    )
    .map_err(map_delaunay)
}

fn map_delaunay(error: ExactFaceDelaunayError) -> ExactFaceRefinementError {
    ExactFaceRefinementError::new(
        ExactFaceRefinementErrorKind::Delaunay(error.kind),
        &error.source_face_id,
        error.reason,
    )
}

fn invalid(pslg: &ExactFacePslg, reason: impl Into<String>) -> ExactFaceRefinementError {
    ExactFaceRefinementError::new(
        ExactFaceRefinementErrorKind::InvalidGeometry,
        &pslg.source_face_id,
        reason,
    )
}
