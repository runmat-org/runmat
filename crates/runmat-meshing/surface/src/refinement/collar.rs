use runmat_meshing_core::{predicate::orient2d, PredicateSign, SurfaceQualityTargets};

use crate::{ExactFaceGeometry, ExactFacePslg, ParametricMetricTensor};

use super::{
    ExactFaceFeatureCollar, ExactFaceFeatureCollars, ExactFaceRefinementError,
    ExactFaceRefinementErrorKind,
};

pub fn derive_exact_face_feature_collars(
    pslg: &ExactFacePslg,
    geometry: &ExactFaceGeometry,
    quality: SurfaceQualityTargets,
) -> Result<ExactFaceFeatureCollars, ExactFaceRefinementError> {
    validate_inputs(pslg, geometry, quality)?;
    let threshold = quality.minimum_metric_angle_degrees.to_radians();
    let mut collars = Vec::new();
    for (loop_index, loop_record) in pslg.loops.iter().enumerate() {
        let segment_indices =
            loop_segments(pslg, loop_record.first_segment, loop_record.segment_count)?;
        let winding = loop_winding(pslg, &segment_indices)?;
        for position in 0..segment_indices.len() {
            let previous_segment =
                segment_indices[(position + segment_indices.len() - 1) % segment_indices.len()];
            let current_segment = segment_indices[position];
            let previous = pslg.segments[previous_segment].vertex_indices[0];
            let current = pslg.segments[current_segment].vertex_indices[0];
            let next = pslg.segments[current_segment].vertex_indices[1];
            let turn = orient2d([
                pslg.vertices[previous as usize].uv,
                pslg.vertices[current as usize].uv,
                pslg.vertices[next as usize].uv,
            ])
            .map_err(|_| invalid(pslg, "feature collar has invalid predicate input"))?;
            let material_is_convex = if loop_index == 0 {
                turn == winding
            } else {
                turn != PredicateSign::Zero && turn != winding
            };
            if !material_is_convex {
                continue;
            }
            let angle = metric_angle(pslg, geometry, previous, current, next)?;
            if angle < threshold {
                let mut incident = [previous_segment as u32, current_segment as u32];
                incident.sort_unstable();
                collars.push(ExactFaceFeatureCollar {
                    pslg_vertex_index: current,
                    incident_segment_indices: incident,
                    feature_angle_rad: angle,
                });
            }
        }
    }
    collars.sort_by_key(|collar| (collar.pslg_vertex_index, collar.incident_segment_indices));
    let result = ExactFaceFeatureCollars {
        source_face_id: pslg.source_face_id.clone(),
        collars,
    };
    validate_exact_face_feature_collars(&result, pslg, geometry, quality)?;
    Ok(result)
}

pub fn validate_exact_face_feature_collars(
    collars: &ExactFaceFeatureCollars,
    pslg: &ExactFacePslg,
    geometry: &ExactFaceGeometry,
    quality: SurfaceQualityTargets,
) -> Result<(), ExactFaceRefinementError> {
    validate_inputs(pslg, geometry, quality)?;
    if collars.source_face_id != pslg.source_face_id
        || collars.collars.windows(2).any(|pair| {
            (pair[0].pslg_vertex_index, pair[0].incident_segment_indices)
                >= (pair[1].pslg_vertex_index, pair[1].incident_segment_indices)
        })
    {
        return Err(invalid(pslg, "feature collar inventory is not canonical"));
    }
    let expected = derive_collar_evidence(pslg, geometry, quality)?;
    if collars.collars.len() != expected.len()
        || collars
            .collars
            .iter()
            .zip(expected)
            .any(|(actual, expected)| {
                actual.pslg_vertex_index != expected.pslg_vertex_index
                    || actual.incident_segment_indices != expected.incident_segment_indices
                    || actual.feature_angle_rad.to_bits() != expected.feature_angle_rad.to_bits()
            })
    {
        return Err(invalid(
            pslg,
            "feature collar evidence differs from independent reconstruction",
        ));
    }
    Ok(())
}

fn derive_collar_evidence(
    pslg: &ExactFacePslg,
    geometry: &ExactFaceGeometry,
    quality: SurfaceQualityTargets,
) -> Result<Vec<ExactFaceFeatureCollar>, ExactFaceRefinementError> {
    let threshold = quality.minimum_metric_angle_degrees.to_radians();
    let mut result = Vec::new();
    for (loop_index, loop_record) in pslg.loops.iter().enumerate() {
        let indices = loop_segments(pslg, loop_record.first_segment, loop_record.segment_count)?;
        let winding = loop_winding(pslg, &indices)?;
        for position in 0..indices.len() {
            let incoming = indices[(position + indices.len() - 1) % indices.len()];
            let outgoing = indices[position];
            let vertices = [
                pslg.segments[incoming].vertex_indices[0],
                pslg.segments[outgoing].vertex_indices[0],
                pslg.segments[outgoing].vertex_indices[1],
            ];
            let turn = orient2d(vertices.map(|index| pslg.vertices[index as usize].uv))
                .map_err(|_| invalid(pslg, "feature collar has invalid predicate input"))?;
            let convex = (loop_index == 0 && turn == winding)
                || (loop_index > 0 && turn != PredicateSign::Zero && turn != winding);
            if !convex {
                continue;
            }
            let angle = metric_angle(pslg, geometry, vertices[0], vertices[1], vertices[2])?;
            if angle < threshold {
                let mut incident_segment_indices = [incoming as u32, outgoing as u32];
                incident_segment_indices.sort_unstable();
                result.push(ExactFaceFeatureCollar {
                    pslg_vertex_index: vertices[1],
                    incident_segment_indices,
                    feature_angle_rad: angle,
                });
            }
        }
    }
    result.sort_by_key(|collar| (collar.pslg_vertex_index, collar.incident_segment_indices));
    Ok(result)
}

fn validate_inputs(
    pslg: &ExactFacePslg,
    geometry: &ExactFaceGeometry,
    quality: SurfaceQualityTargets,
) -> Result<(), ExactFaceRefinementError> {
    quality.validate().map_err(|error| {
        ExactFaceRefinementError::new(
            ExactFaceRefinementErrorKind::InvalidQuality,
            &pslg.source_face_id,
            error.to_string(),
        )
    })?;
    if geometry.source_face_id != pslg.source_face_id
        || geometry.vertices.len() != pslg.vertices.len()
        || geometry.vertices.iter().enumerate().any(|(index, vertex)| {
            vertex.pslg_vertex_index != index as u32
                || vertex.evaluation.uv != pslg.vertices[index].uv
        })
    {
        return Err(invalid(
            pslg,
            "feature collar geometry does not match the PSLG",
        ));
    }
    Ok(())
}

fn loop_segments(
    pslg: &ExactFacePslg,
    first: u32,
    count: u32,
) -> Result<Vec<usize>, ExactFaceRefinementError> {
    let start = first as usize;
    let end = start
        .checked_add(count as usize)
        .ok_or_else(|| invalid(pslg, "feature collar loop range overflow"))?;
    if count < 3 || end > pslg.segments.len() {
        return Err(invalid(pslg, "feature collar loop range is invalid"));
    }
    let segments = &pslg.segments[start..end];
    if segments.iter().any(|segment| {
        segment
            .vertex_indices
            .iter()
            .any(|index| *index as usize >= pslg.vertices.len())
    }) || segments
        .iter()
        .zip(segments.iter().cycle().skip(1))
        .take(segments.len())
        .any(|(left, right)| left.vertex_indices[1] != right.vertex_indices[0])
    {
        return Err(invalid(
            pslg,
            "feature collar loop is disconnected or references an absent vertex",
        ));
    }
    Ok((start..end).collect())
}

fn loop_winding(
    pslg: &ExactFacePslg,
    segments: &[usize],
) -> Result<PredicateSign, ExactFaceRefinementError> {
    let vertices = segments
        .iter()
        .map(|index| pslg.segments[*index].vertex_indices[0])
        .collect::<Vec<_>>();
    let extreme = (0..vertices.len())
        .min_by(|left, right| {
            let left = pslg.vertices[vertices[*left] as usize].uv;
            let right = pslg.vertices[vertices[*right] as usize].uv;
            left[0]
                .total_cmp(&right[0])
                .then_with(|| left[1].total_cmp(&right[1]))
        })
        .ok_or_else(|| invalid(pslg, "feature collar loop is empty"))?;
    for backward in 1..vertices.len() {
        for forward in 1..vertices.len() {
            let previous = vertices[(extreme + vertices.len() - backward) % vertices.len()];
            let next = vertices[(extreme + forward) % vertices.len()];
            if previous == next {
                continue;
            }
            let sign = orient2d([
                pslg.vertices[previous as usize].uv,
                pslg.vertices[vertices[extreme] as usize].uv,
                pslg.vertices[next as usize].uv,
            ])
            .map_err(|_| invalid(pslg, "feature collar has invalid predicate input"))?;
            if sign != PredicateSign::Zero {
                return Ok(sign);
            }
        }
    }
    Err(invalid(pslg, "feature collar loop has zero exact area"))
}

fn metric_angle(
    pslg: &ExactFacePslg,
    geometry: &ExactFaceGeometry,
    previous: u32,
    current: u32,
    next: u32,
) -> Result<f64, ExactFaceRefinementError> {
    let origin = pslg.vertices[current as usize].uv;
    let left = pslg.vertices[previous as usize].uv;
    let right = pslg.vertices[next as usize].uv;
    let metric = geometry.vertices[current as usize].evaluation.sizing_metric;
    let first = [left[0] - origin[0], left[1] - origin[1]];
    let second = [right[0] - origin[0], right[1] - origin[1]];
    let first_squared = metric
        .squared_length(first)
        .map_err(|reason| invalid(pslg, reason))?;
    let second_squared = metric
        .squared_length(second)
        .map_err(|reason| invalid(pslg, reason))?;
    let dot = metric_dot(metric, first, second);
    let cosine = (dot / (first_squared * second_squared).sqrt()).clamp(-1.0, 1.0);
    let angle = cosine.acos();
    if angle.is_finite() && angle > 0.0 {
        Ok(angle)
    } else {
        Err(invalid(pslg, "feature collar angle is invalid"))
    }
}

fn metric_dot(metric: ParametricMetricTensor, left: [f64; 2], right: [f64; 2]) -> f64 {
    left[0] * (metric.uu * right[0] + metric.uv * right[1])
        + left[1] * (metric.uv * right[0] + metric.vv * right[1])
}

fn invalid(pslg: &ExactFacePslg, reason: impl Into<String>) -> ExactFaceRefinementError {
    ExactFaceRefinementError::new(
        ExactFaceRefinementErrorKind::InvalidGeometry,
        &pslg.source_face_id,
        reason,
    )
}
