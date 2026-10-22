//! Decision equivalence and bounded work on repeated independent face regions.
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

fn direct_batch(
    candidates: &[ExactFaceRefinementCandidate],
    geometry: &ExactFaceGeometry,
    pslg: &ExactFacePslg,
    separation: f64,
) -> Vec<ExactFaceRefinementCandidate> {
    let mut selected = Vec::<ExactFaceRefinementCandidate>::new();
    for candidate in candidates {
        let metric = geometry.triangles[candidate.triangle_index as usize].deviation_samples[3]
            .sizing_metric;
        if pslg
            .vertices
            .iter()
            .any(|vertex| select::metric_distance(candidate.uv, vertex.uv, metric) < separation)
        {
            continue;
        }
        if selected.iter().all(|existing| {
            candidate
                .triangle
                .vertex_indices
                .iter()
                .all(|vertex| !existing.triangle.vertex_indices.contains(vertex))
                && select::metric_distance(candidate.uv, existing.uv, metric) >= separation
        }) {
            selected.push(candidate.clone());
        }
    }
    selected
}

struct Count(AtomicUsize);
impl GeometryEvaluationControl for Count {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        self.0.fetch_add(1, Ordering::Relaxed);
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
//   boundary: Indexed batching matches direct canonical decisions under the tested anisotropic
//   metrics and bounds checkpoint work on 1,024 independent regions. This finite spatial-search
//   witness does not establish original CAD convergence or whole-study performance.
#[test]
fn indexed_batches_preserve_direct_anisotropic_decisions_and_local_work() {
    for metric in [
        ParametricMetricTensor {
            uu: 1.0,
            uv: 0.0,
            vv: 1.0,
        },
        ParametricMetricTensor {
            uu: 100.0,
            uv: 0.1,
            vv: 0.01,
        },
        ParametricMetricTensor {
            uu: 1.0,
            uv: 0.999999999,
            vv: 1.0,
        },
    ] {
        let mut geometry = geometry([2.0, 1.0, 1.0], 0.5, 1.0, 0.0, 0.0);
        let mut pslg = triangle_pslg(&geometry);
        let vertices = geometry.vertices.clone();
        let source_vertices = pslg.vertices.clone();
        let triangle = geometry.triangles[0].clone();
        geometry.vertices.clear();
        geometry.triangles.clear();
        pslg.vertices.clear();
        let mut candidates = Vec::new();
        let count = if metric.uv == 0.0 { 1024 } else { 32 };
        for index in 0..count {
            let offset = (index % 32) as f64 * 10.0;
            let y = (index / 32) as f64 * 10.0;
            for (vertex, source) in vertices.iter().zip(&source_vertices) {
                let mut vertex = vertex.clone();
                vertex.evaluation.uv[0] += offset;
                vertex.evaluation.uv[1] += y;
                geometry.vertices.push(vertex);
                let mut source = *source;
                source.uv[0] += offset;
                source.uv[1] += y;
                pslg.vertices.push(source);
            }
            let mut current = triangle.clone();
            current.triangle.vertex_indices = current
                .triangle
                .vertex_indices
                .map(|vertex| vertex + (index * vertices.len()) as u32);
            current.deviation_samples[3].sizing_metric = metric;
            let candidate = ExactFaceRefinementCandidate {
                source_face_id: geometry.source_face_id.clone(),
                triangle_index: index as u32,
                triangle: current.triangle,
                reason: ExactFaceRefinementReason::MetricAngle,
                uv: [offset + 0.4, y + 0.4],
            };
            candidates.push(candidate.clone());
            candidates.push(candidate);
            geometry.triangles.push(current);
        }
        let control = Count(AtomicUsize::new(0));
        let actual = select::independent_candidate_batch(
            candidates.clone(),
            &geometry,
            &pslg,
            0.1,
            Some(&control),
        )
        .unwrap();
        assert_eq!(actual, direct_batch(&candidates, &geometry, &pslg, 0.1));
        if count == 1024 {
            assert_eq!(actual.len(), count);
            assert!(
                control.0.load(Ordering::Relaxed) < count * 500,
                "query work must not grow as candidate-node pairs: {}",
                control.0.load(Ordering::Relaxed)
            );
        }
    }
}

#[test]
fn metric_query_bounds_include_direct_threshold_decisions() {
    use crate::refinement::spatial::metric_bounds;
    for metric in [
        ParametricMetricTensor {
            uu: 1e-8,
            uv: 0.0,
            vv: 1e8,
        },
        ParametricMetricTensor {
            uu: 4.0,
            uv: 1.5,
            vv: 1.0,
        },
        ParametricMetricTensor {
            uu: 1.0,
            uv: 0.999999999999,
            vv: 1.0,
        },
    ] {
        for center in [[0.0, 0.0], [1e12, -1e12]] {
            let bounds = metric_bounds(
                center,
                metric,
                0.1,
                center.into_iter().map(f64::abs).fold(1.0, f64::max),
            );
            for x in -50..=50 {
                for y in -50..=50 {
                    let point = [center[0] + x as f64 * 0.001, center[1] + y as f64 * 0.001];
                    if select::metric_distance(center, point, metric) <= 0.1 {
                        assert!(bounds.contains_point([point[0], point[1], 0.0]));
                    }
                }
            }
        }
    }
}

use std::sync::atomic::AtomicU64;
struct Budget {
    held: AtomicU64,
    peak: AtomicU64,
    maximum: u64,
    probes: AtomicUsize,
    cancel_at: usize,
}
impl GeometryEvaluationControl for Budget {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        if self.probes.fetch_add(1, Ordering::Relaxed) + 1 == self.cancel_at {
            return Err(GeometryEvaluationError::new(
                runmat_geometry_core::GeometryEvaluationErrorKind::Cancelled,
                "query cancelled",
            ));
        }
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
                runmat_geometry_core::GeometryEvaluationErrorKind::AllocationBudgetExceeded,
                "query inventory denied",
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
//   boundary: Query construction admits input, sorting scratch and retained index storage before
//   use; denial preserves its typed cause and success releases temporary storage. Declared storage
//   excludes allocator overhead and measured RSS.
#[test]
fn query_scratch_is_admitted_before_allocation_and_released_after_use() {
    let geometry = geometry([2.0, 1.0, 1.0], 0.5, 1.0, 0.0, 0.0);
    let pslg = triangle_pslg(&geometry);
    for maximum in [1, 1_000_000] {
        let control = Budget {
            held: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            maximum,
            probes: AtomicUsize::new(0),
            cancel_at: usize::MAX,
        };
        let result = select::independent_candidate_batch(
            vec![candidate(geometry.source_face_id.clone(), [0.4, 0.4])],
            &geometry,
            &pslg,
            0.1,
            Some(&control),
        );
        if maximum == 1 {
            assert_eq!(
                result.unwrap_err().kind,
                ExactFaceRefinementErrorKind::Metric(
                    crate::ExactFaceMetricErrorKind::GeometryEvaluation(
                        runmat_geometry_core::GeometryEvaluationErrorKind::AllocationBudgetExceeded
                    )
                )
            );
        } else {
            assert!(result.is_ok());
            assert!(control.peak.load(Ordering::Relaxed) > 0);
        }
        assert_eq!(control.held.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn metric_query_retains_overflow_and_underflow_sensitive_direct_cases() {
    use crate::refinement::spatial::metric_bounds;
    let metric = ParametricMetricTensor {
        uu: 1.0,
        uv: -0.5,
        vv: 1.0,
    };
    assert!(metric_bounds([0.0, 0.0], metric, 0.1, 1e200).contains_point([1e200, 1e200, 0.0]));
    let metric = ParametricMetricTensor {
        uu: 1.0,
        uv: 0.0,
        vv: 1.0,
    };
    assert!(select::metric_distance([0.0, 0.0], [1e-162, 0.0], metric) < 1e-320);
    assert!(metric_bounds([0.0, 0.0], metric, 1e-320, 1.0).contains_point([1e-162, 0.0, 0.0]));
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Interruptions throughout query construction and use release previously admitted
//   indexes and preserve cancellation. This finite ownership witness does not certify whole-task
//   residency.
#[test]
fn interrupted_query_preparation_releases_completed_indexes() {
    let geometry = geometry([2.0, 1.0, 1.0], 0.5, 1.0, 0.0, 0.0);
    let pslg = triangle_pslg(&geometry);
    let mut interrupted_after_storage = false;
    for cancel_at in 1..80 {
        let control = Budget {
            held: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            maximum: 1_000_000,
            probes: AtomicUsize::new(0),
            cancel_at,
        };
        let result = select::independent_candidate_batch(
            vec![candidate(geometry.source_face_id.clone(), [0.4, 0.4])],
            &geometry,
            &pslg,
            0.1,
            Some(&control),
        );
        if let Err(error) = result {
            assert_eq!(
                error.kind,
                ExactFaceRefinementErrorKind::Metric(
                    crate::ExactFaceMetricErrorKind::GeometryEvaluation(
                        runmat_geometry_core::GeometryEvaluationErrorKind::Cancelled
                    )
                )
            );
            interrupted_after_storage |= control.peak.load(Ordering::Relaxed) > 0;
        }
        assert_eq!(control.held.load(Ordering::Relaxed), 0);
    }
    assert!(interrupted_after_storage);
}
