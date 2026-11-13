//! Complete working buffers are admitted before realization; one field owns provenance.
use super::*;
use std::sync::atomic::{AtomicU64, Ordering};
struct Budget {
    held: AtomicU64,
    peak: AtomicU64,
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
        let next = self.held.load(Ordering::Relaxed) + bytes;
        if next > self.maximum {
            return Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::AllocationBudgetExceeded,
                "geometry buffers denied",
            ));
        }
        self.held.store(next, Ordering::Relaxed);
        self.peak.fetch_max(next, Ordering::Relaxed);
        Ok(())
    }
    fn release_allocation_bytes(&self, bytes: u64) {
        assert!(self.held.fetch_sub(bytes, Ordering::Relaxed) >= bytes);
    }
}
// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Complete vertex and triangle geometry buffers are admitted before allocation. Exact
//   requested contiguous capacity accepts, one byte less rejects, and scope exit releases those
//   buffers. Shared provenance is separately field-owned; allocator headers and process RSS are
//   excluded.
#[test]
fn complete_geometry_buffers_admit_exact_requested_storage_and_reject_one_byte_less() {
    let fixture = fixture(None);
    let evaluator = evaluator(&fixture);
    let request = request();
    let field = crate::ResolvedFaceMetricField::new(&fixture.topology, &request).unwrap();
    let exact = crate::face_geometry::storage::buffer_bytes(
        fixture.pslg.vertices.len(),
        fixture.trimmed.triangles.len(),
    )
    .unwrap();
    assert!(exact > fixture.pslg.vertices.len() as u64 * 24);
    for maximum in [exact - 1, exact] {
        let budget = Budget {
            held: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            maximum,
        };
        {
            let scope = runmat_geometry_core::GeometryAllocationScope::new(&budget);
            let mut context =
                ExactFaceGeometryContext::new(&fixture.topology, &request, &evaluator, &scope);
            context.metric_field = Some(&field);
            let result = evaluate_exact_face_geometry_in_parameterization(
                &fixture.trimmed,
                &fixture.pslg,
                &crate::ExactFaceChartParameterization::EvaluatorParameters,
                context,
            );
            if maximum < exact {
                assert_eq!(
                    result.unwrap_err().kind,
                    ExactFaceGeometryErrorKind::Metric(
                        crate::ExactFaceMetricErrorKind::GeometryEvaluation(
                            GeometryEvaluationErrorKind::AllocationBudgetExceeded
                        )
                    )
                );
                assert_eq!(budget.peak.load(Ordering::Relaxed), 0);
            } else {
                let geometry = result.unwrap();
                assert_eq!(
                    crate::face_geometry::storage::buffer_bytes(
                        geometry.vertices.capacity(),
                        geometry.triangles.capacity()
                    ),
                    Some(exact)
                );
                assert_eq!(budget.held.load(Ordering::Relaxed), exact);
                assert!(geometry.triangles.iter().all(|triangle| triangle
                    .deviation_samples
                    .iter()
                    .all(|sample| std::sync::Arc::ptr_eq(
                        &sample.provenance,
                        &geometry.vertices[0].evaluation.provenance
                    ))));
            }
        }
        assert_eq!(budget.held.load(Ordering::Relaxed), 0);
    }
}
