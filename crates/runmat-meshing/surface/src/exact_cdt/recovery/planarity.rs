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
) -> Result<u64, ExactFaceDelaunayError> {
    let mut ordered = edges.keys().copied().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        minimum(left, 0, pslg)
            .total_cmp(&minimum(right, 0, pslg))
            .then_with(|| left.cmp(right))
    });
    let mut predicates = 0u64;
    for left_index in 0..ordered.len() {
        if cancellation.is_cancelled() {
            return Err(error(
                pslg,
                ExactFaceDelaunayErrorKind::Cancelled,
                "planarity validation cancelled",
            ));
        }
        let left = ordered[left_index];
        let left_maximum_x = maximum(&left, 0, pslg);
        for right in &ordered[left_index + 1..] {
            if minimum(right, 0, pslg) > left_maximum_x {
                break;
            }
            if !bounds_overlap(&left, right, pslg) {
                continue;
            }
            predicates = consume(predicates, 4, options, pslg)?;
            if !segments_intersect(left, *right, pslg)? {
                continue;
            }
            predicates = consume(predicates, 2, options, pslg)?;
            if !intersection_is_shared_endpoint(left, *right, pslg)? {
                return Err(error(
                    pslg,
                    ExactFaceDelaunayErrorKind::InvalidTopology,
                    format!("triangulation edges {left:?} and {right:?} intersect"),
                ));
            }
        }
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
