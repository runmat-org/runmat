//! Returned claims are checked against measured original points, never replayed.
use super::*;
use crate::builtins::math::optim::common::constrained_numeric as numeric;
use numeric::{Geometry, Numeric};
pub(super) async fn validate(
    returned: crate::execution::RootedValueSequence,
    actual: usize,
    base: &Point,
    initial: f64,
    best: &mut Option<Best>,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<()> {
    let returned =
        super::super::super::evaluation::select(returned, 4, evaluator.cooperation.runtime())?;
    let mut returned = {
        let _entry = evaluator
            .cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        returned.try_into_values().map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })?
    };
    let success = match returned.next().expect("selected success") {
        Value::Bool(value) => value,
        Value::LogicalArray(value) if value.data.len() == 1 => value.data[0] != 0,
        _ => {
            return Err(invalid(
                "custom SearchFcn successSearch must be a logical scalar",
            ))
        }
    };
    let x = returned.next().expect("selected x");
    let f = returned.next().expect("selected f");
    let count = returned.next().expect("selected count");
    // All required result roles are admitted before any provider transfer;
    // unsuccessful xBest/fBest payloads remain dormant.
    for (value, label, geometry, needed) in [
        (&count, "custom search funccount", Geometry::Scalar, true),
        (&x, "custom search xBest", Geometry::Vector, success),
        (&f, "custom search fBest", Geometry::Scalar, success),
    ] {
        if needed
            && !numeric::preflight(
                value,
                label,
                geometry,
                &evaluator.context.policy,
                evaluator.cooperation.runtime(),
            )?
        {
            return Err(invalid(
                "custom SearchFcn required results must be real numeric",
            ));
        }
    }
    let Numeric::Real { values, .. } = numeric::read(
        count,
        "custom search funccount",
        Geometry::Scalar,
        &evaluator.context.policy,
        evaluator.cooperation,
    )
    .await?
    else {
        return Err(invalid("custom SearchFcn funccount must be finite"));
    };
    if values[0] != actual as f64 {
        return Err(invalid(
            "custom SearchFcn funccount differs from actual objective attempts",
        ));
    }
    if !success {
        *best = None;
        return Ok(());
    }
    let Numeric::Real { values: x, .. } = numeric::read(
        x,
        "custom search xBest",
        Geometry::Vector,
        &evaluator.context.policy,
        evaluator.cooperation,
    )
    .await?
    else {
        return Err(invalid("custom SearchFcn xBest must be finite"));
    };
    let Numeric::Real { values: f, .. } = numeric::read(
        f,
        "custom search fBest",
        Geometry::Scalar,
        &evaluator.context.policy,
        evaluator.cooperation,
    )
    .await?
    else {
        return Err(invalid("custom SearchFcn fBest must be finite"));
    };
    let (observed_x, observed_f) = best
        .as_ref()
        .map_or((&base.x, initial), |best| (&best.point.x, best.score));
    if x != *observed_x || f[0] != observed_f {
        return Err(invalid(
            "custom SearchFcn must return its actually measured best point and merit",
        ));
    }
    Ok(())
}
