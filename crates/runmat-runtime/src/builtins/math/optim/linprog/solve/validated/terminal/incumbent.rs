//! Original stopped or numerical-failure incumbent check.
use super::*;
use crate::builtins::math::optim::linprog::compiled;
pub(super) async fn run<V: compiled::Validation>(
    prepared: Prepared<'_, V>,
    flag: i32,
    message: &'static str,
) -> BuiltinResult<LinprogOutcome> {
    let Prepared {
        model,
        mut point,
        cooperation,
        controls,
        validation,
        ..
    } = prepared;
    super::super::incumbent(
        &model,
        std::mem::take(&mut point.x),
        flag,
        controls.iterations(),
        message,
        cooperation,
        validation,
    )
    .await
}
