use runmat_meshing_core::{predicate::orient2d, PredicateSign};

use super::{
    ExactFaceBoundary, ExactFaceBoundarySegment, ExactSurfaceBoundaryConflict,
    ExactSurfaceBoundaryError, ExactSurfaceBoundaryErrorKind,
};

const MAX_PSLG_PAIR_TESTS: usize = 10_000_000;

pub(super) fn validate_face_segment_intersections(
    face: &ExactFaceBoundary,
) -> Result<(), ExactSurfaceBoundaryError> {
    let mut segments = std::iter::once(&face.outer_loop)
        .chain(&face.inner_loops)
        .flat_map(|loop_boundary| &loop_boundary.segments)
        .collect::<Vec<_>>();
    segments.sort_by(|left, right| {
        minimum_x(left)
            .total_cmp(&minimum_x(right))
            .then_with(|| left.source_edge_id.cmp(&right.source_edge_id))
            .then_with(|| left.source_coedge_id.cmp(&right.source_coedge_id))
            .then_with(|| left.node_ids.cmp(&right.node_ids))
    });

    let mut pair_tests = 0usize;
    for left_index in 0..segments.len() {
        let left = segments[left_index];
        let left_maximum_x = maximum_x(left);
        for right in &segments[left_index + 1..] {
            if minimum_x(right) > left_maximum_x {
                break;
            }
            if !bounds_overlap(left, right) {
                continue;
            }
            pair_tests = pair_tests.saturating_add(1);
            if pair_tests > MAX_PSLG_PAIR_TESTS {
                return Err(ExactSurfaceBoundaryError::new(
                    ExactSurfaceBoundaryErrorKind::ResourceLimit,
                    Some(face.source_face_id.clone()),
                    "face PSLG intersection search exceeds its hard pair bound",
                ));
            }
            if segments_intersect(left.node_uv, right.node_uv)?
                && !intersection_is_declared_endpoint(left, right)?
            {
                return Err(ExactSurfaceBoundaryError::new(
                    ExactSurfaceBoundaryErrorKind::InvalidPslg,
                    Some(face.source_face_id.clone()),
                    "undeclared face-local trim segment intersection",
                )
                .with_conflict(ExactSurfaceBoundaryConflict {
                    source_edge_ids: [left.source_edge_id.clone(), right.source_edge_id.clone()],
                    segment_uv: [left.node_uv, right.node_uv],
                }));
            }
        }
    }
    Ok(())
}

fn segments_intersect(
    left: [[f64; 2]; 2],
    right: [[f64; 2]; 2],
) -> Result<bool, ExactSurfaceBoundaryError> {
    let signs = [
        orient2d([left[0], left[1], right[0]]),
        orient2d([left[0], left[1], right[1]]),
        orient2d([right[0], right[1], left[0]]),
        orient2d([right[0], right[1], left[1]]),
    ];
    let signs = signs
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| {
            ExactSurfaceBoundaryError::new(
                ExactSurfaceBoundaryErrorKind::InvalidContract,
                None,
                "face PSLG predicate received an invalid coordinate",
            )
        })?;
    if opposite(signs[0], signs[1]) && opposite(signs[2], signs[3]) {
        return Ok(true);
    }
    Ok(
        (signs[0] == PredicateSign::Zero && within_bounds(right[0], left))
            || (signs[1] == PredicateSign::Zero && within_bounds(right[1], left))
            || (signs[2] == PredicateSign::Zero && within_bounds(left[0], right))
            || (signs[3] == PredicateSign::Zero && within_bounds(left[1], right)),
    )
}

fn opposite(left: PredicateSign, right: PredicateSign) -> bool {
    matches!(
        (left, right),
        (PredicateSign::Negative, PredicateSign::Positive)
            | (PredicateSign::Positive, PredicateSign::Negative)
    )
}

fn within_bounds(point: [f64; 2], segment: [[f64; 2]; 2]) -> bool {
    (0..2).all(|axis| {
        point[axis] >= segment[0][axis].min(segment[1][axis])
            && point[axis] <= segment[0][axis].max(segment[1][axis])
    })
}

fn minimum_x(segment: &ExactFaceBoundarySegment) -> f64 {
    segment.node_uv[0][0].min(segment.node_uv[1][0])
}

fn maximum_x(segment: &ExactFaceBoundarySegment) -> f64 {
    segment.node_uv[0][0].max(segment.node_uv[1][0])
}

fn bounds_overlap(left: &ExactFaceBoundarySegment, right: &ExactFaceBoundarySegment) -> bool {
    (0..2).all(|axis| {
        left.node_uv[0][axis].min(left.node_uv[1][axis])
            <= right.node_uv[0][axis].max(right.node_uv[1][axis])
            && right.node_uv[0][axis].min(right.node_uv[1][axis])
                <= left.node_uv[0][axis].max(left.node_uv[1][axis])
    })
}

fn intersection_is_declared_endpoint(
    left: &ExactFaceBoundarySegment,
    right: &ExactFaceBoundarySegment,
) -> Result<bool, ExactSurfaceBoundaryError> {
    for left_index in 0..2 {
        for right_index in 0..2 {
            if left.node_ids[left_index] != right.node_ids[right_index]
                || left.node_uv[left_index] != right.node_uv[right_index]
            {
                continue;
            }
            let left_other = left.node_uv[1 - left_index];
            let right_other = right.node_uv[1 - right_index];
            let right_other_on_left = segments_intersect(
                [left.node_uv[left_index], left_other],
                [right_other, right_other],
            )?;
            let left_other_on_right = segments_intersect(
                [right.node_uv[right_index], right_other],
                [left_other, left_other],
            )?;
            return Ok(!left_other_on_right && !right_other_on_left);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
    use runmat_meshing_core::StableDigest;

    use super::*;
    use crate::{ExactFaceBoundaryLoop, ExactFaceBoundarySegment};

    #[test]
    fn crossing_trim_segments_report_both_edges_and_uv_witnesses() {
        let face = ExactFaceBoundary {
            source_face_id: id(PersistentEntityKind::Face, "face"),
            outer_loop: ExactFaceBoundaryLoop {
                source_wire_id: id(PersistentEntityKind::Wire, "wire"),
                orientation: runmat_geometry_core::TopologicalOrientation::Forward,
                segments: vec![
                    segment("a", 1, 2, [[0.0, 0.0], [1.0, 1.0]]),
                    segment("b", 3, 4, [[0.0, 1.0], [1.0, 0.0]]),
                ],
            },
            inner_loops: Vec::new(),
        };

        let error = validate_face_segment_intersections(&face).unwrap_err();
        assert_eq!(error.kind, ExactSurfaceBoundaryErrorKind::InvalidPslg);
        let conflict = error.conflict.unwrap();
        assert_eq!(conflict.source_edge_ids[0].source_topology_id, "a");
        assert_eq!(conflict.source_edge_ids[1].source_topology_id, "b");
        assert_eq!(conflict.segment_uv[1], [[0.0, 1.0], [1.0, 0.0]]);
    }

    #[test]
    fn declared_shared_vertex_is_not_misclassified_as_a_crossing() {
        let left = segment("a", 1, 2, [[0.0, 0.0], [1.0, 0.0]]);
        let mut right = segment("b", 2, 3, [[1.0, 0.0], [1.0, 1.0]]);
        right.node_ids[0] = left.node_ids[1];
        let face = ExactFaceBoundary {
            source_face_id: id(PersistentEntityKind::Face, "face"),
            outer_loop: ExactFaceBoundaryLoop {
                source_wire_id: id(PersistentEntityKind::Wire, "wire"),
                orientation: runmat_geometry_core::TopologicalOrientation::Forward,
                segments: vec![left, right],
            },
            inner_loops: Vec::new(),
        };
        validate_face_segment_intersections(&face).unwrap();
    }

    fn segment(edge: &str, start: u8, end: u8, node_uv: [[f64; 2]; 2]) -> ExactFaceBoundarySegment {
        ExactFaceBoundarySegment {
            source_coedge_id: id(PersistentEntityKind::Coedge, edge),
            source_edge_id: id(PersistentEntityKind::Edge, edge),
            node_ids: [
                StableDigest::from_bytes([start; 32]),
                StableDigest::from_bytes([end; 32]),
            ],
            node_uv,
        }
    }

    fn id(kind: PersistentEntityKind, name: &str) -> PersistentEntityId {
        PersistentEntityId {
            kind,
            source_topology_id: name.into(),
            assembly_path: vec!["root".into()],
        }
    }
}
