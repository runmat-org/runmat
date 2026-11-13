//! Select an original numerical controller without retaining its constructor frame.
use super::*;
use futures::future::Either;
use std::future::Future;

// Only construction crosses this boundary; the returned future polls normally
// in the existing admitted operation. No extra operation or allocation is added.
#[inline(never)]
pub(super) fn select<'geometry, 'parameters, 'controls, 'evaluator, 'source>(
    algorithm: OptimAlgorithm,
    x: Vec<f64>,
    geometry: &'geometry Geometry,
    parameters: &'parameters Parameters<'controls>,
    evaluator: &'evaluator mut Evaluator<'source>,
) -> impl Future<Output = BuiltinResult<NumericalOutcome>>
       + use<'geometry, 'parameters, 'controls, 'evaluator, 'source> {
    match algorithm {
        OptimAlgorithm::InteriorPoint => Either::Left(super::super::interior_point::minimize(
            x, geometry, parameters, evaluator,
        )),
        OptimAlgorithm::Sqp => Either::Right(Either::Left(super::super::sqp::minimize(
            x, geometry, parameters, evaluator,
        ))),
        OptimAlgorithm::ActiveSet => {
            Either::Right(Either::Right(super::super::sqp::minimize_with(
                x,
                geometry,
                parameters,
                evaluator,
                super::super::sqp::Strategy::WorkingSet,
            )))
        }
        _ => unreachable!("implemented modeled algorithm"),
    }
}
