//! A cold partition prepares one producer field; independent validation remains separate.
use super::*;
use std::sync::atomic::{AtomicU64, Ordering};
struct Counter(AtomicU64);
impl GeometryEvaluationControl for Counter {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        if std::path::Path::new(std::panic::Location::caller().file())
            .ends_with(std::path::Path::new("exact_metric/resolved/preparation.rs"))
        {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
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
//   boundary: A complete cold exact-face partition prepares its producer metric field once across
//   geometry and acceptance. Independent validation reconstructs its own input-derived reference;
//   this fixture does not establish complete original CAD convergence or whole-study scalability.
#[test]
fn one_partition_reuses_one_prepared_producer_field_across_geometry_and_acceptance() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("exact fixture")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let curves = discretize_shared_curves(
        &topology,
        &evaluator,
        &evaluator,
        &UniformCurveMetric::from_target_size_m(0.5).unwrap(),
        &Control,
        curve_options(),
    )
    .unwrap();
    let request = MetricFieldRequest {
        entity_supports: Vec::new(),
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(100.0).unwrap(),
        maximum_grading_ratio: 2.0,
        contributions: vec![],
        cells_across_gap: None,
    };
    let expected = Counter(AtomicU64::new(0));
    crate::ResolvedFaceMetricField::new_with_control(&topology, &request, &expected).unwrap();
    assert!(expected.0.load(Ordering::Relaxed) > 0);
    let actual = Counter(AtomicU64::new(0));
    let partition = face_partition_descriptors(&topology, 1).unwrap().remove(0);
    let result = mesh_exact_face_partition(
        partition,
        ExactFacePartitionContext {
            topology: &topology,
            curves: &curves,
            metric_request: &request,
            quality: permissive_quality(),
            evaluator: &evaluator,
            geometry_control: &actual,
            cancellation: &NeverCancelled,
        },
        ExactFacePartitionOptions::default(),
    )
    .unwrap();
    assert!(matches!(
        result.outcome,
        ExactFacePartitionOutcome::Converged { .. }
    ));
    assert_eq!(
        actual.0.load(Ordering::Relaxed),
        expected.0.load(Ordering::Relaxed)
    );
}
