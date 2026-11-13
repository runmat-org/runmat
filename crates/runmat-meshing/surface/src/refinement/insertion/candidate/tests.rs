use super::*;
use crate::{
    ExactFaceDelaunayTriangle, ExactFacePslgSegment, ExactFacePslgSegmentSource,
    ExactFacePslgVertex,
};
use runmat_geometry_core::{
    GeometryEvaluationControl, GeometryEvaluationError, PersistentEntityId, PersistentEntityKind,
};
use runmat_meshing_core::{NeverCancelled, StableDigest};
struct Control;
impl GeometryEvaluationControl for Control {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_search_work(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_allocation_bytes(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
}
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Batch insertion admission preserves direct node duplication, protected-segment and
//   trimmed-domain decisions on the tested disconnected regions. Existing robust predicates remain
//   authoritative.
#[test]
fn indexed_membership_matches_direct_admission_in_disconnected_trimmed_regions() {
    let face = PersistentEntityId::new(PersistentEntityKind::Face, "query-fixture".into(), vec![]);
    let triangles = vec![
        ExactFaceDelaunayTriangle {
            vertex_indices: [0, 1, 2],
        },
        ExactFaceDelaunayTriangle {
            vertex_indices: [3, 4, 5],
        },
    ];
    let pslg = ExactFacePslg {
        source_face_id: face.clone(),
        vertices: [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [2.0, 1.0],
        ]
        .into_iter()
        .enumerate()
        .map(|(index, uv)| ExactFacePslgVertex {
            node_id: StableDigest::from_bytes([index as u8 + 1; 32]),
            seam_image: None,
            uv,
        })
        .collect(),
        segments: [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]]
            .into_iter()
            .map(|vertices| ExactFacePslgSegment {
                source: ExactFacePslgSegmentSource::ChartCut {
                    cut_id: StableDigest::from_bytes([7; 32]),
                },
                vertex_indices: vertices,
                edge_parameters: None,
            })
            .collect(),
        loops: vec![],
    };
    let trimmed = ExactFaceTrimmedDelaunay {
        source_face_id: face.clone(),
        triangles: triangles.clone(),
        boundary_segments: vec![],
        removed_exterior_triangle_count: 0,
        removed_hole_triangle_count: 0,
    };
    let mut successes = 0;
    for x in -4..=16 {
        for y in -4..=8 {
            let candidate = ExactFaceRefinementCandidate {
                source_face_id: face.clone(),
                triangle_index: 0,
                triangle: triangles[0],
                reason: crate::ExactFaceRefinementReason::PhysicalAspectRatio,
                uv: [x as f64 * 0.25, y as f64 * 0.25],
            };
            let expected = validate_candidate(&pslg, &trimmed, &candidate, &NeverCancelled);
            let actual = validate_candidates(
                &pslg,
                &trimmed,
                &[candidate],
                &NeverCancelled,
                Some(&Control),
            );
            successes += usize::from(expected.is_ok());
            assert_eq!(actual, expected);
        }
    }
    assert!(successes > 0);
}
