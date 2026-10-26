//! A shared kernel proposes; this adapter alone publishes original least-squares results.
use super::*;
use conic::solve::{Candidate, StopKind};
use model::Model;
pub(super) async fn solve(
    problem: Problem,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Outcome> {
    let products =
        compiled::Products::for_output_count(crate::output_count::current_output_count());
    solve_validated(
        problem,
        products,
        objective::Units::HalfSquared,
        cooperation,
        &mut compiled::Supplied,
    )?
    .await
}

mod interior;
mod selection;
pub(super) fn solve_validated<'a, V: compiled::Validation + 'a>(
    problem: Problem,
    products: compiled::Products,
    units: objective::Units,
    cooperation: &'a mut ScalarCooperation,
    validation: &'a mut V,
) -> BuiltinResult<crate::execution::RuntimeContinuationFuture<'a, Outcome>> {
    let runtime = cooperation.runtime().cloned();
    crate::execution::admit_borrowed_operation_with_contextual_error(runtime.as_ref(), move || {
        crate::execution::after_result(
            selection::prepare(problem, products, units, cooperation, validation),
            selection::execute,
        )
    })
}
