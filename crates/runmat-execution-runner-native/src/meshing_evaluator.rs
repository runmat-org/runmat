use runmat_geometry_core::{GeometryContractError, GeometryEvaluationControl};
use runmat_meshing_execution::{
    ExactMeshingEvaluatorProvider, ExactMeshingGeometryEvaluation, MeshingKernelDispatcher,
    PortableMeshingEvaluatorProvider, PreparedExactGeometryObjects,
};

/// Selects the evaluator implementation declared by an admitted exact-geometry closure.
///
/// A kernel representation is an explicit OCCT capability requirement. Its absence denotes a
/// fully portable evaluator registry; neither case is attempted as a fallback for the other.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeMeshingEvaluatorProvider;

impl ExactMeshingEvaluatorProvider for NativeMeshingEvaluatorProvider {
    fn evaluator<'a>(
        &self,
        geometry: &'a PreparedExactGeometryObjects,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<Box<dyn ExactMeshingGeometryEvaluation + 'a>, GeometryContractError> {
        match geometry.kernel_representation.as_deref() {
            Some(representation) => kernel_evaluator(representation, geometry, control),
            None => PortableMeshingEvaluatorProvider.evaluator(geometry, control),
        }
    }
}

#[cfg(feature = "occt-native")]
fn kernel_evaluator<'a>(
    representation: &[u8],
    geometry: &'a PreparedExactGeometryObjects,
    control: &dyn GeometryEvaluationControl,
) -> Result<Box<dyn ExactMeshingGeometryEvaluation + 'a>, GeometryContractError> {
    runmat_geometry_io::OcctExactEvaluator::from_closure_controlled(
        representation,
        geometry.document.source.meters_per_source_unit,
        &geometry.topology,
        &geometry.evaluators,
        control,
    )
    .map(|evaluator| Box::new(evaluator) as Box<dyn ExactMeshingGeometryEvaluation>)
    .map_err(|error| GeometryContractError::invalid("native exact evaluator", error.to_string()))
}

#[cfg(not(feature = "occt-native"))]
fn kernel_evaluator<'a>(
    _representation: &[u8],
    _geometry: &'a PreparedExactGeometryObjects,
    _control: &dyn GeometryEvaluationControl,
) -> Result<Box<dyn ExactMeshingGeometryEvaluation + 'a>, GeometryContractError> {
    Err(GeometryContractError::invalid(
        "native exact evaluator",
        "kernel-backed exact geometry requires an OCCT-capable native worker",
    ))
}

pub fn native_meshing_kernel_dispatcher() -> MeshingKernelDispatcher<NativeMeshingEvaluatorProvider>
{
    MeshingKernelDispatcher::new(NativeMeshingEvaluatorProvider)
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
            &ExactCadImportOptions::default(),
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
        assert!(point.into_iter().all(f64::is_finite));
    }
}
