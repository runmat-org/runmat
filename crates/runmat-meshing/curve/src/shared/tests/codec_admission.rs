//! Original canonical bytes must survive admitted production; denial and
//! cancellation must release scratch and keep the original failure category.
use super::*;
use runmat_geometry_core::{
    GeometryAllocationScope, GeometryEvaluationControl, GeometryEvaluationError,
    GeometryEvaluationErrorKind,
};
use runmat_meshing_core::NeverCancelled;
struct Budget {
    live: AtomicU64,
    maximum: u64,
}
impl GeometryEvaluationControl for Budget {
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
        self.live
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
                live.checked_add(bytes).filter(|sum| *sum <= self.maximum)
            })
            .map(|_| ())
            .map_err(|_| {
                GeometryEvaluationError::new(
                    GeometryEvaluationErrorKind::AllocationBudgetExceeded,
                    "curve codec fixture budget",
                )
            })
    }
    fn release_allocation_bytes(&self, bytes: u64) {
        self.live
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
                live.checked_sub(bytes)
            })
            .expect("balanced curve codec ownership");
    }
}
// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Curve and partition encoders preserve original canonical bytes, reject denied
//   workspace before byte production, retain actual output capacity, and release temporary storage.
//   The fixture does not establish native execution or arbitrary CAD capacity.
#[test]
fn admitted_curve_and_partition_encoders_preserve_bytes_and_release_storage() {
    let (_, topology, _) = runmat_geometry_fixtures::exact_circle();
    let mesh = circle_mesh(&topology);
    let batch = SharedCurveBatch {
        schema_version: SHARED_CURVE_BATCH_SCHEMA_VERSION,
        partition: curve_partition_descriptors(&topology, 1).unwrap().remove(0),
        edges: mesh.edges.clone(),
    };
    for maximum in [1, 1024 * 1024] {
        let budget = Budget {
            live: AtomicU64::new(0),
            maximum,
        };
        for batch_route in [false, true] {
            {
                let scope = GeometryAllocationScope::new(&budget);
                let encoded = if batch_route {
                    encode_shared_curve_batch_admitted(&batch, &topology, &NeverCancelled, &scope)
                } else {
                    encode_shared_curve_mesh_admitted(&mesh, &topology, &NeverCancelled, &scope)
                };
                if maximum == 1 {
                    assert_eq!(
                        encoded.unwrap_err().kind,
                        SharedCurveErrorKind::GeometryEvaluation(
                            GeometryEvaluationErrorKind::AllocationBudgetExceeded
                        )
                    );
                } else {
                    let encoded = encoded.unwrap();
                    let original = if batch_route {
                        encode_shared_curve_batch(&batch, &topology).unwrap()
                    } else {
                        encode_shared_curve_mesh(&mesh, &topology).unwrap()
                    };
                    assert_eq!(encoded, original);
                    assert_eq!(
                        budget.live.load(Ordering::Relaxed),
                        encoded.capacity() as u64
                    );
                }
            }
            assert_eq!(budget.live.load(Ordering::Relaxed), 0);
        }
    }
}
#[test]
fn admitted_curve_encoding_preserves_cooperative_cancellation_category() {
    let (_, topology, _) = runmat_geometry_fixtures::exact_circle();
    let mesh = circle_mesh(&topology);
    let budget = Budget {
        live: AtomicU64::new(0),
        maximum: 1024 * 1024,
    };
    for limit in [0, 4, 30] {
        let signal = super::cancellation::CancelAfter {
            probes: AtomicU64::new(0),
            limit,
        };
        {
            let scope = GeometryAllocationScope::new(&budget);
            let error =
                encode_shared_curve_mesh_admitted(&mesh, &topology, &signal, &scope).unwrap_err();
            assert_eq!(
                error.kind,
                SharedCurveErrorKind::GeometryEvaluation(GeometryEvaluationErrorKind::Cancelled)
            );
        }
        assert_eq!(budget.live.load(Ordering::Relaxed), 0);
    }
}
