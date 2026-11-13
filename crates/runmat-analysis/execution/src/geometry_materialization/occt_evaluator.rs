//! Host-selected evaluation of an admitted exact geometry closure.
use runmat_geometry_core::{GeometryContractError, GeometryEvaluationControl};
use runmat_meshing_execution::{
    ExactMeshingEvaluatorProvider, ExactMeshingGeometryEvaluation,
    PortableMeshingEvaluatorProvider, PreparedExactGeometryObjects,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct OcctMeshingEvaluatorProvider;
impl ExactMeshingEvaluatorProvider for OcctMeshingEvaluatorProvider {
    fn evaluator<'a>(
        &self,
        geometry: &'a PreparedExactGeometryObjects,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<Box<dyn ExactMeshingGeometryEvaluation + 'a>, GeometryContractError> {
        match geometry
            .kernel_representation()
            .map_err(|error| GeometryContractError::invalid("exact geometry", error.to_string()))?
        {
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
        geometry
            .document
            .source
            .representation_units
            .meters_per_unit(),
        &geometry.topology,
        &geometry.evaluators,
        control,
    )
    .map(|evaluator| Box::new(evaluator) as Box<dyn ExactMeshingGeometryEvaluation>)
    .map_err(|error| GeometryContractError::invalid("exact evaluator", error.to_string()))
}
#[cfg(not(feature = "occt-native"))]
fn kernel_evaluator<'a>(
    _: &[u8],
    _: &'a PreparedExactGeometryObjects,
    _: &dyn GeometryEvaluationControl,
) -> Result<Box<dyn ExactMeshingGeometryEvaluation + 'a>, GeometryContractError> {
    Err(GeometryContractError::invalid(
        "exact evaluator",
        "This execution host does not provide the admitted exact CAD kernel.",
    ))
}
