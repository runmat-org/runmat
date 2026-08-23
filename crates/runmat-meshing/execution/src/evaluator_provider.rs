use runmat_geometry_core::{GeometryContractError, GeometryModel, PortableExactEvaluator};
use runmat_meshing_tetrahedron::cdt::DelaunayExactEvaluator;

use crate::PreparedExactGeometryObjects;

/// Reconstructs the exact evaluator admitted by a geometry closure.
///
/// The provider is a host-composition port: meshing selects which queries it needs, while the
/// execution host selects the implementation capable of evaluating the closure's kernel ABI.
pub trait ExactMeshingEvaluatorProvider: Send + Sync {
    fn evaluator<'a>(
        &self,
        geometry: &'a PreparedExactGeometryObjects,
    ) -> Result<Box<dyn DelaunayExactEvaluator + 'a>, GeometryContractError>;
}

impl<T> ExactMeshingEvaluatorProvider for &T
where
    T: ExactMeshingEvaluatorProvider + ?Sized,
{
    fn evaluator<'a>(
        &self,
        geometry: &'a PreparedExactGeometryObjects,
    ) -> Result<Box<dyn DelaunayExactEvaluator + 'a>, GeometryContractError> {
        (**self).evaluator(geometry)
    }
}

/// Executes geometry closures whose evaluator registry is fully portable.
#[derive(Clone, Copy, Debug, Default)]
pub struct PortableMeshingEvaluatorProvider;

impl ExactMeshingEvaluatorProvider for PortableMeshingEvaluatorProvider {
    fn evaluator<'a>(
        &self,
        geometry: &'a PreparedExactGeometryObjects,
    ) -> Result<Box<dyn DelaunayExactEvaluator + 'a>, GeometryContractError> {
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
