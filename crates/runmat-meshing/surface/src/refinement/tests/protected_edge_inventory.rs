//! Large protected inventories preserve decisions and charge their temporary index before use.
use super::*;
use runmat_geometry_core::GeometryEvaluationErrorKind;
use std::sync::atomic::{AtomicU64, Ordering};

struct InventoryControl {
    bytes: AtomicU64,
    deny: bool,
}
impl GeometryEvaluationControl for InventoryControl {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_search_work(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_allocation_bytes(&self, bytes: u64) -> Result<(), GeometryEvaluationError> {
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
        if self.deny {
            Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::AllocationBudgetExceeded,
                "boundary index denied",
            ))
        } else {
            Ok(())
        }
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A 16384-entry reversed protected-segment inventory preserves minimum-edge protection
//   and charges its temporary index before use; denied allocation rejects. This component
//   lookup/admission case does not prove arbitrary constrained triangulation, native throughput, or
//   whole-study memory.
#[test]
fn selection_preserves_protection_with_a_large_reversed_segment_inventory() {
    let geometry = geometry([0.005, 0.5, 0.5], 0.001, 1_000.0, 0.0, 0.0);
    let mut pslg = triangle_pslg(&geometry);
    pslg.segments = (0..16_384)
        .map(|index| ExactFacePslgSegment {
            source: ExactFacePslgSegmentSource::ExactTrim {
                source_coedge_id: id(PersistentEntityKind::Coedge, "inventory-coedge"),
                source_edge_id: id(PersistentEntityKind::Edge, "inventory-edge"),
            },
            vertex_indices: [[1, 0], [2, 1], [0, 2]][index % 3],
            edge_parameters: Some([0.0, 1.0]),
        })
        .collect();
    let collars = empty_collars(&geometry);
    for deny in [false, true] {
        let control = InventoryControl {
            bytes: AtomicU64::new(0),
            deny,
        };
        let result = super::super::select::select_exact_face_refinement_candidates(
            &geometry,
            &pslg,
            &collars,
            quality(),
            Some(&control),
        );
        assert_eq!(control.bytes.load(Ordering::Relaxed), 16_384 * 8);
        if deny {
            assert_eq!(
                result.unwrap_err().kind,
                ExactFaceRefinementErrorKind::Metric(
                    crate::ExactFaceMetricErrorKind::GeometryEvaluation(
                        GeometryEvaluationErrorKind::AllocationBudgetExceeded
                    )
                )
            );
        } else {
            assert!(result.unwrap().is_empty());
        }
    }
}
