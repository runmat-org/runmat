//! Selected LSQ defaults reuse the existing field-origin owner.
use super::*;
pub(super) fn normalize(
    runtime: Option<&crate::context::RuntimeContext>,
    options: &mut OptionState,
) -> BuiltinResult<()> {
    use state::OptionOrigin;
    use OptimOptionField::*;
    let algorithm = options
        .get(Algorithm)
        .filter(|v| !matches!(v,Value::Tensor(t)if t.is_empty()))
        .map(|value| algorithm_value(Solver::Lsqlin, value))
        .transpose()?
        .unwrap_or(OptimAlgorithm::InteriorPoint);
    if algorithm == OptimAlgorithm::ActiveSet {
        if options.origin(MaxIter) == Some(OptionOrigin::Default) {
            options.set(MaxIter, Value::Num(2000.));
        }
        for (field, default) in [(TolX, 1e-8), (ObjectiveLimit, -1e20)] {
            if options
                .get(field)
                .is_none_or(|v| matches!(v,Value::Tensor(t)if t.is_empty()))
            {
                options.set(field, Value::Num(default));
            }
        }
        return Ok(());
    }
    if algorithm != OptimAlgorithm::TrustRegionReflective {
        return Ok(());
    }
    let legacy = options
        .get(TolFun)
        .filter(|v| !matches!(v,Value::Tensor(t)if t.is_empty()))
        .cloned();
    let origin = options.origin(TolFun);
    for field in [OptimalityTolerance, FunctionTolerance] {
        let empty = options
            .get(field)
            .is_none_or(|v| matches!(v,Value::Tensor(t)if t.is_empty()));
        if empty {
            if legacy.is_some() && origin.is_some_and(|origin| origin != OptionOrigin::Default) {
                options.set_with_origin(
                    field,
                    legacy.clone().expect("active legacy value"),
                    origin.expect("active legacy origin"),
                );
            } else {
                options.set(field, Value::Num(100. * f64::EPSILON));
            }
        } else if let Some(value) = &legacy {
            let modern_origin = options.origin(field);
            if origin == Some(OptionOrigin::Explicit)
                && modern_origin == origin
                && options.get(field) != Some(value)
            {
                return Err(optimoptions_error_with(
                    &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                    format!("optimoptions: conflicting TolFun and {}", field.name()),
                ));
            }
        }
    }
    options.remove(TolFun);
    for (field, value) in [
        (TolPCG, Value::Num(0.1)),
        (PrecondBandWidth, Value::Num(0.)),
        (SubproblemAlgorithm, state::text(runtime, "cg")?),
    ] {
        if options
            .get(field)
            .is_none_or(|v| matches!(v,Value::Tensor(t)if t.is_empty()))
        {
            options.set(field, value);
        }
    }
    for field in [MaxPCGIter, TypicalX] {
        if options.get(field).is_none() {
            options.set(
                field,
                crate::builtins::math::optim::common::output_callback::empty(runtime)?,
            );
        }
    }
    Ok(())
}
