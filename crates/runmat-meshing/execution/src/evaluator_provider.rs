use runmat_geometry_core::{
    GeometryContractError, GeometryEvaluationControl, GeometryModel, PortableExactEvaluator,
};
use runmat_meshing_tetrahedron::cdt::DelaunayExactEvaluator;

use crate::PreparedExactGeometryObjects;

pub trait ExactMeshingGeometryEvaluation:
    DelaunayExactEvaluator + runmat_meshing_curve::ExactFacePcurveEvaluator
{
}

impl<T> ExactMeshingGeometryEvaluation for T where T: DelaunayExactEvaluator {}

/// Reconstructs the exact evaluator admitted by a geometry closure.
///
/// The provider is a host-composition port: meshing selects which queries it needs, while the
/// execution host selects the implementation capable of evaluating the closure's kernel ABI.
pub trait ExactMeshingEvaluatorProvider: Send + Sync {
    /// Resolves geometry-induced sizing without changing authored request identity or controls.
    fn geometry_metric(
        &self,
        geometry: &PreparedExactGeometryObjects,
        evaluator: &dyn ExactMeshingGeometryEvaluation,
        control: &dyn GeometryEvaluationControl,
        request: &runmat_meshing_core::MeshingRequest,
    ) -> Result<
        std::sync::Arc<runmat_meshing_core::MetricFieldRequest>,
        runmat_meshing_curve::SharedCurveError,
    > {
        runmat_meshing_curve::derive_curve_geometry_metric(
            &geometry.topology,
            evaluator,
            control,
            &request.metric,
            request.quality.curve,
            request.quality.surface,
        )
        .map(std::sync::Arc::new)
    }

    fn evaluator<'a>(
        &self,
        geometry: &'a PreparedExactGeometryObjects,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<Box<dyn ExactMeshingGeometryEvaluation + 'a>, GeometryContractError>;
}

impl<T> ExactMeshingEvaluatorProvider for &T
where
    T: ExactMeshingEvaluatorProvider + ?Sized,
{
    fn geometry_metric(
        &self,
        geometry: &PreparedExactGeometryObjects,
        evaluator: &dyn ExactMeshingGeometryEvaluation,
        control: &dyn GeometryEvaluationControl,
        request: &runmat_meshing_core::MeshingRequest,
    ) -> Result<
        std::sync::Arc<runmat_meshing_core::MetricFieldRequest>,
        runmat_meshing_curve::SharedCurveError,
    > {
        (**self).geometry_metric(geometry, evaluator, control, request)
    }
    fn evaluator<'a>(
        &self,
        geometry: &'a PreparedExactGeometryObjects,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<Box<dyn ExactMeshingGeometryEvaluation + 'a>, GeometryContractError> {
        (**self).evaluator(geometry, control)
    }
}

/// Executes geometry closures whose evaluator registry is fully portable.
#[derive(Clone, Copy, Debug, Default)]
pub struct PortableMeshingEvaluatorProvider;

impl ExactMeshingEvaluatorProvider for PortableMeshingEvaluatorProvider {
    fn evaluator<'a>(
        &self,
        geometry: &'a PreparedExactGeometryObjects,
        _control: &dyn GeometryEvaluationControl,
    ) -> Result<Box<dyn ExactMeshingGeometryEvaluation + 'a>, GeometryContractError> {
        let GeometryModel::ExactBRep { model } = &geometry.document.model else {
            return Err(GeometryContractError::invalid(
                "meshing evaluator geometry",
                "exact meshing evaluation requires exact B-rep geometry",
            ));
        };
        Ok(Box::new(PortableExactEvaluator::new(
            &geometry.evaluators,
            &geometry.topology,
            model,
        )?))
    }
}
