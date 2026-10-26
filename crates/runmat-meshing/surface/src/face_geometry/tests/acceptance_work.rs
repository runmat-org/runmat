//! Impossible adaptive work is rejected before independent evaluation or allocation.
use super::*;
use std::sync::atomic::{AtomicU64, Ordering};
struct ObservedWork(AtomicU64);
impl GeometryEvaluationControl for ObservedWork {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        self.checkpoint()
    }
    fn consume_search_work(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        self.checkpoint()
    }
    fn consume_allocation_bytes(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        self.checkpoint()
    }
}
// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: An adaptive sampling demand exceeding the explicit cap rejects before independent
//   geometry evaluation or report allocation. Overflow rejects conservatively. The lower bound
//   never authorizes skipping required samples or deeper quality-driven refinement.
#[test]
fn insufficient_adaptive_sample_budget_rejects_before_geometry_work() {
    let fixture = fixture(None);
    let evaluator = evaluator(&fixture);
    let request = request();
    let geometry = evaluate_exact_face_geometry(
        &fixture.trimmed,
        &fixture.pslg,
        &fixture.topology,
        &request,
        &evaluator,
        &Control,
    )
    .unwrap();
    let options = crate::ExactFaceAcceptanceOptions::default();
    let minimum = options
        .minimum_sample_count(geometry.triangles.len() as u64)
        .unwrap();
    assert!(minimum > 0);
    let mesh = crate::ExactFaceRefinedMesh {
        topology: crate::ExactFaceRefinedTopology {
            pslg: fixture.pslg.clone(),
            constrained: fixture.constrained.clone(),
            trimmed: fixture.trimmed.clone(),
        },
        feature_collars: crate::ExactFaceFeatureCollars {
            source_face_id: geometry.source_face_id.clone(),
            collars: vec![],
        },
        geometry,
        interior_insertion_count: 0,
    };
    let observed = ObservedWork(AtomicU64::new(0));
    let failure = crate::accept_exact_face_mesh(
        &mesh,
        crate::ExactFaceRefinementContext::new(
            &fixture.topology,
            &request,
            &evaluator,
            &observed,
            &NeverCancelled,
        ),
        runmat_meshing_core::SurfaceQualityTargets {
            minimum_metric_edge_length: 0.01,
            minimum_metric_angle_degrees: 0.1,
            maximum_physical_aspect_ratio: 1000.0,
            maximum_chordal_deviation_m: runmat_types::physical::Length::new(0.0001),
            maximum_normal_deviation_degrees: 180.0,
        },
        crate::ExactFaceAcceptanceOptions {
            maximum_samples: minimum - 1,
            ..options
        },
    )
    .unwrap_err();
    assert_eq!(
        failure.kind,
        crate::ExactFaceAcceptanceErrorKind::ResourceLimit
    );
    assert!(failure.reason.contains("minimum adaptive"));
    assert_eq!(observed.0.load(Ordering::Relaxed), 0);
    assert!(crate::ExactFaceAcceptanceOptions {
        minimum_subdivision_depth: 32,
        ..options
    }
    .minimum_sample_count(u64::MAX)
    .is_none());
}
