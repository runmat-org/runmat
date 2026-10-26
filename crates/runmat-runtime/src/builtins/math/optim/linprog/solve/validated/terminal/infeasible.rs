//! Original selected status arithmetic and callback ordering.
use super::*;
use crate::builtins::math::optim::linprog::compiled;
pub(super) async fn run<V: compiled::Validation>(
    prepared: Prepared<'_, V>,
) -> BuiltinResult<LinprogOutcome> {
    let Prepared {
        model,
        point,
        cooperation,
        controls,
        validation,
        ..
    } = prepared;
    if compiled::farkas(&model, &point.z, cooperation, validation).await? {
        Ok(LinprogOutcome::failure(
            -2,
            controls.iterations(),
            "Problem is infeasible; original constraint certificate verified.",
        ))
    } else {
        Ok(LinprogOutcome::failure(
            -4,
            controls.iterations(),
            "Infeasibility certificate could not be verified against original constraints.",
        ))
    }
}
