use runmat_meshing_core::StableDigest;
use sha2::{Digest, Sha256};

use crate::exact_cdt::{build_canonical_pslg, PslgLoopInput, PslgSegmentInput};
use crate::{
    ExactFaceBoundary, ExactFaceChartError, ExactFaceChartErrorKind, ExactFaceChartOptions,
    ExactFacePslg, ExactFacePslgLoopSource, ExactFacePslgSegmentSource, ExactFacePslgVertex,
};

pub(super) fn build_periodic_annulus_pslg(
    boundary: &mut ExactFaceBoundary,
    windings: &[[i32; 2]],
    periodicity: [Option<f64>; 2],
    chart_id: StableDigest,
    options: ExactFaceChartOptions,
) -> Result<ExactFacePslg, ExactFaceChartError> {
    let periodic_axes = (0..2)
        .filter(|axis| windings.iter().any(|winding| winding[*axis] != 0))
        .collect::<Vec<_>>();
    if boundary.inner_loops.len() != 1
        || windings.len() != 2
        || periodic_axes.len() != 1
        || windings.iter().any(|winding| {
            winding
                .iter()
                .any(|component| !(-1..=1).contains(component))
        })
    {
        return Err(requires_partition(
            boundary,
            "periodic face requires a bounded multi-chart partition",
        ));
    }
    let axis = periodic_axes[0];
    if windings[0][axis] + windings[1][axis] != 0
        || windings[0][axis].abs() != 1
        || (0..2).any(|other| other != axis && windings.iter().any(|w| w[other] != 0))
    {
        return Err(requires_partition(
            boundary,
            "periodic boundary winding is not one balanced annulus",
        ));
    }
    let period = periodicity[axis].ok_or_else(|| {
        ExactFaceChartError::new(
            ExactFaceChartErrorKind::InvalidInput,
            &boundary.source_face_id,
            "annulus winding uses a nonperiodic chart axis",
        )
    })?;
    align_inner_loop(boundary, axis, period, options)?;

    let outer = &boundary.outer_loop;
    let inner = &boundary.inner_loops[0];
    let outer_start = endpoint(outer, 0, 0);
    let outer_end = endpoint(outer, outer.segments.len() - 1, 1);
    let inner_start = endpoint(inner, 0, 0);
    let inner_end = endpoint(inner, inner.segments.len() - 1, 1);
    let cut_id = chart_cut_id(chart_id);
    let mut segments = physical_segments(outer).collect::<Vec<_>>();
    segments.push(PslgSegmentInput {
        source: ExactFacePslgSegmentSource::ChartCut { cut_id },
        endpoints: [outer_end, inner_start],
        edge_parameters: None,
    });
    segments.extend(physical_segments(inner));
    segments.push(PslgSegmentInput {
        source: ExactFacePslgSegmentSource::ChartCut { cut_id },
        endpoints: [inner_end, outer_start],
        edge_parameters: None,
    });
    let pslg = build_canonical_pslg(
        &boundary.source_face_id,
        vec![PslgLoopInput {
            source: ExactFacePslgLoopSource::ChartBoundary {
                boundary_id: chart_id,
            },
            orientation: outer.orientation,
            segments,
        }],
    )
    .map_err(|failure| {
        ExactFaceChartError::new(
            ExactFaceChartErrorKind::InvalidInput,
            &boundary.source_face_id,
            failure.to_string(),
        )
    })?;
    validate_periodic_annulus_pslg(&pslg, boundary, chart_id, cut_id, periodicity, options)?;
    Ok(pslg)
}

fn validate_periodic_annulus_pslg(
    pslg: &ExactFacePslg,
    boundary: &ExactFaceBoundary,
    chart_id: StableDigest,
    cut_id: StableDigest,
    periodicity: [Option<f64>; 2],
    options: ExactFaceChartOptions,
) -> Result<(), ExactFaceChartError> {
    let physical_count =
        boundary.outer_loop.segments.len() + boundary.inner_loops[0].segments.len();
    if pslg.source_face_id != boundary.source_face_id
        || pslg.loops.len() != 1
        || pslg.loops[0].source
            != (ExactFacePslgLoopSource::ChartBoundary {
                boundary_id: chart_id,
            })
        || pslg.loops[0].first_segment != 0
        || pslg.loops[0].segment_count as usize != physical_count + 2
        || pslg.segments.len() != physical_count + 2
        || pslg.vertices.is_empty()
    {
        return Err(invalid(
            boundary,
            "annulus chart PSLG inventory is inconsistent",
        ));
    }
    for (left, right) in pslg
        .segments
        .iter()
        .zip(pslg.segments.iter().cycle().skip(1))
    {
        if left.vertex_indices[1] != right.vertex_indices[0]
            || left
                .vertex_indices
                .iter()
                .any(|index| *index as usize >= pslg.vertices.len())
        {
            return Err(invalid(
                boundary,
                "annulus chart PSLG is not one closed loop",
            ));
        }
    }
    let outer_count = boundary.outer_loop.segments.len();
    validate_physical_range(pslg, 0, &boundary.outer_loop.segments, boundary)?;
    validate_physical_range(
        pslg,
        outer_count + 1,
        &boundary.inner_loops[0].segments,
        boundary,
    )?;
    let cuts = [&pslg.segments[outer_count], pslg.segments.last().unwrap()];
    if cuts.iter().any(|segment| {
        segment.source != ExactFacePslgSegmentSource::ChartCut { cut_id }
            || segment.edge_parameters.is_some()
            || segment.vertex_indices[0] == segment.vertex_indices[1]
    }) {
        return Err(invalid(
            boundary,
            "annulus chart cut provenance is inconsistent",
        ));
    }
    let first = cuts[0]
        .vertex_indices
        .map(|index| pslg.vertices[index as usize]);
    let second = cuts[1]
        .vertex_indices
        .map(|index| pslg.vertices[index as usize]);
    if [first[0].node_id, first[1].node_id] != [second[1].node_id, second[0].node_id] {
        return Err(invalid(
            boundary,
            "annulus chart cut images do not share reversed 3D endpoint identities",
        ));
    }
    for (left, right) in [(first[0], second[1]), (first[1], second[0])] {
        for (axis, period) in periodicity.into_iter().enumerate() {
            let residual = left.uv[axis] - right.uv[axis];
            let tolerance = options.maximum_periodic_residual
                * left.uv[axis].abs().max(right.uv[axis].abs()).max(1.0);
            let valid = match period {
                Some(period) => {
                    let shifts = (residual / period).round();
                    shifts.abs() <= options.maximum_period_shifts as f64
                        && (residual - shifts * period).abs() <= tolerance
                }
                None => residual.abs() <= tolerance,
            };
            if !valid {
                return Err(invalid(
                    boundary,
                    "annulus chart cut images are not periodic equivalents",
                ));
            }
        }
    }
    Ok(())
}

fn validate_physical_range(
    pslg: &ExactFacePslg,
    offset: usize,
    expected: &[crate::ExactFaceBoundarySegment],
    boundary: &ExactFaceBoundary,
) -> Result<(), ExactFaceChartError> {
    for (actual, expected) in pslg.segments[offset..offset + expected.len()]
        .iter()
        .zip(expected)
    {
        let source = ExactFacePslgSegmentSource::ExactTrim {
            source_coedge_id: expected.source_coedge_id.clone(),
            source_edge_id: expected.source_edge_id.clone(),
        };
        let endpoints = actual
            .vertex_indices
            .map(|index| pslg.vertices[index as usize]);
        if actual.source != source
            || actual.edge_parameters != Some(expected.edge_parameters)
            || endpoints.map(|vertex| vertex.node_id) != expected.node_ids
            || endpoints.map(|vertex| vertex.seam_image) != [expected.seam_image; 2]
            || endpoints.map(|vertex| vertex.uv) != expected.node_uv
        {
            return Err(invalid(
                boundary,
                "annulus chart physical segment differs from its exact trim",
            ));
        }
    }
    Ok(())
}

fn align_inner_loop(
    boundary: &mut ExactFaceBoundary,
    axis: usize,
    period: f64,
    options: ExactFaceChartOptions,
) -> Result<(), ExactFaceChartError> {
    let outer_end = boundary.outer_loop.segments.last().unwrap().node_uv[1][axis];
    let inner_start = boundary.inner_loops[0].segments[0].node_uv[0][axis];
    let shift = ((outer_end - inner_start) / period).round();
    if !shift.is_finite() || shift.abs() > options.maximum_period_shifts as f64 {
        return Err(ExactFaceChartError::new(
            ExactFaceChartErrorKind::InvalidInput,
            &boundary.source_face_id,
            "annulus chart alignment exceeds its hard period-shift bound",
        ));
    }
    let offset = shift * period;
    for segment in &mut boundary.inner_loops[0].segments {
        for uv in &mut segment.node_uv {
            uv[axis] += offset;
            if !uv[axis].is_finite() {
                return Err(ExactFaceChartError::new(
                    ExactFaceChartErrorKind::InvalidInput,
                    &boundary.source_face_id,
                    "annulus chart alignment produced a nonfinite coordinate",
                ));
            }
        }
    }
    Ok(())
}

fn physical_segments(
    boundary: &crate::ExactFaceBoundaryLoop,
) -> impl Iterator<Item = PslgSegmentInput> + '_ {
    boundary.segments.iter().map(|segment| PslgSegmentInput {
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
}

fn endpoint(
    boundary: &crate::ExactFaceBoundaryLoop,
    segment: usize,
    endpoint: usize,
) -> ExactFacePslgVertex {
    let value = &boundary.segments[segment];
    ExactFacePslgVertex {
        node_id: value.node_ids[endpoint],
        seam_image: value.seam_image,
        uv: value.node_uv[endpoint],
    }
}

fn chart_cut_id(chart_id: StableDigest) -> StableDigest {
    let mut digest = Sha256::new();
    digest.update(b"runmat.exact-face-chart-cut\0");
    digest.update(1u16.to_be_bytes());
    digest.update(chart_id.bytes());
    digest.update(0u32.to_be_bytes());
    StableDigest::from_bytes(digest.finalize().into())
}

fn requires_partition(boundary: &ExactFaceBoundary, reason: &str) -> ExactFaceChartError {
    ExactFaceChartError::new(
        ExactFaceChartErrorKind::RequiresMultipleCharts,
        &boundary.source_face_id,
        reason,
    )
}

fn invalid(boundary: &ExactFaceBoundary, reason: &str) -> ExactFaceChartError {
    ExactFaceChartError::new(
        ExactFaceChartErrorKind::InvalidInput,
        &boundary.source_face_id,
        reason,
    )
}
