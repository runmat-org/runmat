//! Desktop uses the same exact evaluator composition as the browser sidecar.
pub use runmat_analysis_execution::OcctMeshingEvaluatorProvider as NativeMeshingEvaluatorProvider;
use runmat_meshing_execution::MeshingKernelDispatcher;

pub fn native_meshing_kernel_dispatcher() -> MeshingKernelDispatcher<NativeMeshingEvaluatorProvider>
{
    let dispatcher = MeshingKernelDispatcher::new(NativeMeshingEvaluatorProvider);
    match crate::meshing_subtask_cache::native_subtask_cache() {
        Some(cache) => dispatcher.with_subtask_cache(cache),
        None => dispatcher,
    }
}

#[cfg(all(test, feature = "occt-native"))]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use runmat_execution_artifact::object::ObjectInventoryLimits;
    use runmat_geometry_core::{
        ExactCurveEvaluator, GeometryEvaluationControl, GeometryEvaluationError,
    };
    use runmat_geometry_io::{
        import::GeometryImportContext, import_exact_cad, ExactCadImportOptions, GeometryFormat,
    };
    use runmat_meshing_execution::{prepare_exact_geometry_objects, ExactMeshingEvaluatorProvider};

    use super::NativeMeshingEvaluatorProvider;

    const BOX: &[u8] = include_bytes!("../../runmat-geometry/io/tests/fixtures/box.brep");

    #[derive(Default)]
    struct Unlimited {
        checkpoints: AtomicU64,
    }

    impl GeometryEvaluationControl for Unlimited {
        fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
            self.checkpoints.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn consume_iterations(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }

        fn consume_search_work(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }

        fn consume_allocation_bytes(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
    }

    #[test]
    fn native_provider_executes_a_transferred_occt_closure() {
        let imported = import_exact_cad(
            "box.brep",
            BOX,
            GeometryFormat::Brep,
            &ExactCadImportOptions {
                source_units: Some(runmat_geometry_core::LengthUnit::Meter),
                source_units_authority: runmat_geometry_io::SourceUnitsAuthority::Authored,
                ..Default::default()
            },
            &GeometryImportContext::new(),
        )
        .unwrap();
        let closure = imported.build_closure().unwrap();
        let geometry = prepare_exact_geometry_objects(
            closure.document,
            imported.topology,
            imported.evaluators,
            Some(imported.representation),
            imported.healing_report,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let control = Unlimited::default();
        let evaluator = NativeMeshingEvaluatorProvider
            .evaluator(&geometry, &control)
            .unwrap();
        assert!(control.checkpoints.load(Ordering::Relaxed) >= 3);
        let curve_id = &geometry.topology.edges[0].curve_evaluator_id;
        let range = ExactCurveEvaluator::parameter_range(evaluator.as_ref(), curve_id).unwrap();
        let point = ExactCurveEvaluator::point(evaluator.as_ref(), curve_id, range.start, &control)
            .unwrap();
        assert!(point
            .canonical_values()
            .iter()
            .all(|value| value.is_finite()));
    }
}
