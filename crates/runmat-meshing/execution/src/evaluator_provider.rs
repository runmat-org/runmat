use runmat_geometry_core::{
    GeometryContractError, GeometryEvaluationControl, GeometryModel, PortableExactEvaluator,
};
use runmat_meshing_tetrahedron::cdt::DelaunayExactEvaluator;

use crate::PreparedExactGeometryObjects;

pub trait ExactMeshingGeometryEvaluation: DelaunayExactEvaluator {}

impl<T> ExactMeshingGeometryEvaluation for T where T: DelaunayExactEvaluator {}

/// Reconstructs the exact evaluator admitted by a geometry closure.
///
/// The provider is a host-composition port: meshing selects which queries it needs, while the
/// execution host selects the implementation capable of evaluating the closure's kernel ABI.
pub trait ExactMeshingEvaluatorProvider: Send + Sync {
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
