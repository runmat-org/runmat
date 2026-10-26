//! Domain view over the shared original-row arithmetic.
use super::Linearization;
use crate::builtins::math::optim::common::{working_set, ScalarCooperation};
use crate::BuiltinResult;
use runmat_builtins::BuiltinCatalogIdentity;
pub(super) fn feasible<'a>(
    model: &'a Linearization,
    x: &'a [f64],
    tolerance: f64,
    owner: BuiltinCatalogIdentity,
    c: &'a mut ScalarCooperation,
) -> impl std::future::Future<Output = BuiltinResult<bool>> + 'a {
    working_set::rows::feasible(
        working_set::Rows {
            a: &model.a,
            b: &model.b,
            equalities: model.equalities,
        },
        x,
        tolerance,
        owner,
        c,
    )
}
