use std::collections::BTreeMap;

use runmat_meshing_core::{predicate::orient2d, MeshingCancellationSignal, PredicateSign};

use crate::{
    ExactFaceDelaunayError, ExactFaceDelaunayErrorKind, ExactFaceDelaunayOptions, ExactFacePslg,
};

use crate::exact_cdt::topology::EdgeUse;

pub(super) fn validate_planar_edges(
    edges: &BTreeMap<[u32; 2], Vec<EdgeUse>>,
    pslg: &ExactFacePslg,
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
    geometry_control: Option<&dyn runmat_geometry_core::GeometryEvaluationControl>,
) -> Result<u64, ExactFaceDelaunayError> {
    use runmat_geometry_core::{GeometryAllocationScope, GeometryEvaluationControl};
    use runmat_meshing_core::quality::spatial_index::{
        Aabb3, PackedBoundsSpatialIndex, SpatialEntry,
    };
    checkpoint(cancellation, pslg)?;
    let storage = geometry_control.map(GeometryAllocationScope::new);
    let profile =
        PackedBoundsSpatialIndex::<[u32; 2]>::storage_profile(edges.len()).ok_or_else(|| {
            error(
                pslg,
                ExactFaceDelaunayErrorKind::ElementLimit,
                "planarity index storage overflow",
            )
        })?;
    let input_bytes = edges
        .len()
        .checked_mul(std::mem::size_of::<SpatialEntry<[u32; 2]>>())
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| {
            error(
                pslg,
                ExactFaceDelaunayErrorKind::ElementLimit,
                "planarity entry storage overflow",
            )
        })?;
    let scratch = input_bytes
        .checked_add(profile.construction_scratch_bytes)
        .ok_or_else(|| {
            error(
                pslg,
                ExactFaceDelaunayErrorKind::ElementLimit,
                "planarity scratch storage overflow",
            )
        })?;
    let total = scratch.checked_add(profile.retained_bytes).ok_or_else(|| {
        error(
            pslg,
            ExactFaceDelaunayErrorKind::ElementLimit,
            "planarity storage overflow",
        )
    })?;
    let map_control = |failure: runmat_geometry_core::GeometryEvaluationError| {
        error(
            pslg,
            ExactFaceDelaunayErrorKind::GeometryEvaluation(failure.kind),
            failure.reason,
        )
    };
    if let Some(storage) = &storage {
        storage
            .consume_allocation_bytes(total)
            .map_err(map_control)?;
    }
    let edge_bounds = |edge: [u32; 2]| Aabb3 {
        min_m: [minimum(&edge, 0, pslg), minimum(&edge, 1, pslg), 0.0],
        max_m: [maximum(&edge, 0, pslg), maximum(&edge, 1, pslg), 0.0],
    };
    let entries = edges
        .keys()
        .map(|edge| SpatialEntry {
            bounds: edge_bounds(*edge),
            payload: *edge,
        })
        .collect();
    let mut probes = 0u32;
    let index = PackedBoundsSpatialIndex::from_entries_with_checkpoint(entries, &mut || {
        sampled_checkpoint(&mut probes, cancellation, pslg)
    })?
    .ok_or_else(|| {
        error(
            pslg,
            ExactFaceDelaunayErrorKind::InvalidTopology,
            "planarity index bounds invalid",
        )
    })?;
    if let Some(storage) = &storage {
        storage.release_allocation_bytes(scratch);
    }
    let mut predicates = 0;
    for left in edges.keys() {
        checkpoint(cancellation, pslg)?;
        index.try_visit_bounds_with_checkpoint(
            edge_bounds(*left),
            &mut |entry| {
                let right = entry.payload;
                if right <= *left || !bounds_overlap(left, &right, pslg) {
                    return Ok(());
                }
                predicates = consume(predicates, 4, options, pslg)?;
                if segments_intersect(*left, right, pslg)? {
                    predicates = consume(predicates, 2, options, pslg)?;
                    if !intersection_is_shared_endpoint(*left, right, pslg)? {
                        return Err(error(
                            pslg,
                            ExactFaceDelaunayErrorKind::InvalidTopology,
                            format!("triangulation edges {left:?} and {right:?} intersect"),
                        ));
                    }
                }
                Ok(())
            },
            &mut || sampled_checkpoint(&mut probes, cancellation, pslg),
        )?;
    }
    Ok(predicates)
}

fn segments_intersect(
    left: [u32; 2],
    right: [u32; 2],
    pslg: &ExactFacePslg,
) -> Result<bool, ExactFaceDelaunayError> {
    let left_uv = left.map(|index| pslg.vertices[index as usize].uv);
    let right_uv = right.map(|index| pslg.vertices[index as usize].uv);
    let signs = [
        orientation(left_uv[0], left_uv[1], right_uv[0], pslg)?,
        orientation(left_uv[0], left_uv[1], right_uv[1], pslg)?,
        orientation(right_uv[0], right_uv[1], left_uv[0], pslg)?,
        orientation(right_uv[0], right_uv[1], left_uv[1], pslg)?,
    ];
    if opposite(signs[0], signs[1]) && opposite(signs[2], signs[3]) {
        return Ok(true);
    }
    Ok(
        (signs[0] == PredicateSign::Zero && within(right_uv[0], left_uv))
            || (signs[1] == PredicateSign::Zero && within(right_uv[1], left_uv))
            || (signs[2] == PredicateSign::Zero && within(left_uv[0], right_uv))
            || (signs[3] == PredicateSign::Zero && within(left_uv[1], right_uv)),
    )
}

fn intersection_is_shared_endpoint(
    left: [u32; 2],
    right: [u32; 2],
    pslg: &ExactFacePslg,
) -> Result<bool, ExactFaceDelaunayError> {
    for left_position in 0..2 {
        for right_position in 0..2 {
            if left[left_position] != right[right_position] {
                continue;
            }
            let shared = pslg.vertices[left[left_position] as usize].uv;
            let left_other = pslg.vertices[left[1 - left_position] as usize].uv;
            let right_other = pslg.vertices[right[1 - right_position] as usize].uv;
            let right_on_left = orientation(shared, left_other, right_other, pslg)?
                == PredicateSign::Zero
                && within(right_other, [shared, left_other]);
            let left_on_right = orientation(shared, right_other, left_other, pslg)?
                == PredicateSign::Zero
                && within(left_other, [shared, right_other]);
            return Ok(!right_on_left && !left_on_right);
        }
    }
    Ok(false)
}

fn orientation(
    first: [f64; 2],
    second: [f64; 2],
    third: [f64; 2],
    pslg: &ExactFacePslg,
) -> Result<PredicateSign, ExactFaceDelaunayError> {
    orient2d([first, second, third]).map_err(|predicate| {
        error(
            pslg,
            ExactFaceDelaunayErrorKind::InvalidTopology,
            format!("invalid planarity predicate input: {predicate:?}"),
        )
    })
}

fn bounds_overlap(left: &[u32; 2], right: &[u32; 2], pslg: &ExactFacePslg) -> bool {
    (0..2).all(|axis| {
        minimum(left, axis, pslg) <= maximum(right, axis, pslg)
            && minimum(right, axis, pslg) <= maximum(left, axis, pslg)
    })
}

fn minimum(edge: &[u32; 2], axis: usize, pslg: &ExactFacePslg) -> f64 {
    pslg.vertices[edge[0] as usize].uv[axis].min(pslg.vertices[edge[1] as usize].uv[axis])
}

fn maximum(edge: &[u32; 2], axis: usize, pslg: &ExactFacePslg) -> f64 {
    pslg.vertices[edge[0] as usize].uv[axis].max(pslg.vertices[edge[1] as usize].uv[axis])
}

fn within(point: [f64; 2], segment: [[f64; 2]; 2]) -> bool {
    (0..2).all(|axis| {
        point[axis] >= segment[0][axis].min(segment[1][axis])
            && point[axis] <= segment[0][axis].max(segment[1][axis])
    })
}

fn opposite(left: PredicateSign, right: PredicateSign) -> bool {
    matches!(
        (left, right),
        (PredicateSign::Negative, PredicateSign::Positive)
            | (PredicateSign::Positive, PredicateSign::Negative)
    )
}

fn consume(
    count: u64,
    amount: u64,
    options: ExactFaceDelaunayOptions,
    pslg: &ExactFacePslg,
) -> Result<u64, ExactFaceDelaunayError> {
    let count = count.saturating_add(amount);
    if count > options.maximum_predicate_evaluations {
        Err(error(
            pslg,
            ExactFaceDelaunayErrorKind::SearchWorkLimit,
            "planarity predicate hard limit exceeded",
        ))
    } else {
        Ok(count)
    }
}

fn error(
    pslg: &ExactFacePslg,
    kind: ExactFaceDelaunayErrorKind,
    reason: impl Into<String>,
) -> ExactFaceDelaunayError {
    ExactFaceDelaunayError::new(kind, &pslg.source_face_id, reason)
}

/// Index callbacks fire for every visited node, including empty subtrees. Each edge query
/// already starts with a full probe; inside a query, probe cancellation every 256 callbacks.
/// A probe reads the execution clock, and per-node probes dominated large face validations.
fn sampled_checkpoint(
    probes: &mut u32,
    cancellation: &dyn MeshingCancellationSignal,
    pslg: &ExactFacePslg,
) -> Result<(), ExactFaceDelaunayError> {
    *probes = probes.wrapping_add(1);
    if (*probes).is_multiple_of(256) {
        checkpoint(cancellation, pslg)
    } else {
        Ok(())
    }
}

#[track_caller]
fn checkpoint(
    cancellation: &dyn MeshingCancellationSignal,
    pslg: &ExactFacePslg,
) -> Result<(), ExactFaceDelaunayError> {
    if cancellation.is_cancelled() {
        return Err(error(
            pslg,
            ExactFaceDelaunayErrorKind::Cancelled,
            "planarity validation cancelled",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "planarity/tests.rs"]
mod tests;
