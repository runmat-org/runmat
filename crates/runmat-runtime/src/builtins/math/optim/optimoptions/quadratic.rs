//! Selected QP defaults reuse canonical field origin and typed algorithm.
use super::*;
pub(super) fn normalize(
    runtime: Option<&crate::context::RuntimeContext>,
    options: &mut OptionState,
) -> BuiltinResult<()> {
    use OptimOptionField::*;
    let algorithm = options
        .get(Algorithm)
        .filter(|v| !matches!(v,Value::Tensor(t) if t.is_empty()))
        .map(|v| algorithm_value(Solver::Quadprog, v))
        .transpose()?
        .unwrap_or(OptimAlgorithm::InteriorPointConvex);
    if algorithm == OptimAlgorithm::ActiveSet {
        if options.origin(MaxIter) == Some(state::OptionOrigin::Default) {
            options.set(
                MaxIter,
                crate::builtins::math::optim::common::output_callback::empty(runtime)?,
            );
        }
        for (field, default) in [
            (TolX, 1e-8),
            (ObjectiveLimit, -1e20),
            (MaxTime, f64::INFINITY),
        ] {
            if options
                .get(field)
                .is_none_or(|v| matches!(v,Value::Tensor(t) if t.is_empty()))
            {
                options.set(field, Value::Num(default));
            }
        }
    } else {
        for field in [TolX, ObjectiveLimit, MaxTime] {
            if options
                .get(field)
                .is_some_and(|v| !matches!(v,Value::Tensor(t) if t.is_empty()))
            {
                return Err(optimoptions_error_with(
                    &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                    "optimoptions: StepTolerance, ObjectiveLimit and MaxTime require active-set quadprog",
                ));
            }
        }
    }
    Ok(())
}
