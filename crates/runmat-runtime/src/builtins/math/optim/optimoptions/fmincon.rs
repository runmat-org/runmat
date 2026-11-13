//! Final algorithm selection owns defaults and the legacy TolFun criterion.
use super::*;
pub(super) fn normalize(
    runtime: Option<&crate::context::RuntimeContext>,
    state: &mut OptionState,
) -> BuiltinResult<()> {
    use state::OptionOrigin;
    use OptimOptionField::*;
    let algorithm = state
        .get(Algorithm)
        .filter(|value| !matches!(value,Value::Tensor(t)if t.is_empty()))
        .map(|value| algorithm_value(Solver::Fmincon, value))
        .transpose()?
        .unwrap_or(OptimAlgorithm::InteriorPoint);
    let defaults = default_options(runtime, Solver::Fmincon)?;
    for (key, value) in defaults.fields() {
        if let OptionKey::Field(field) = key {
            if state
                .get(field)
                .is_some_and(|value| matches!(value,Value::Tensor(t)if t.is_empty()))
            {
                state.set(field, value.clone());
            }
        }
    }
    if matches!(
        algorithm,
        OptimAlgorithm::Sqp | OptimAlgorithm::ActiveSet | OptimAlgorithm::TrustRegionReflective
    ) {
        for (field, value) in [
            (MaxIter, Value::Num(400.0)),
            (TolX, Value::Num(1e-6)),
            (
                MaxFunEvals,
                crate::builtins::math::optim::common::output_callback::empty(runtime)?,
            ),
        ] {
            if state.origin(field) == Some(OptionOrigin::Default) {
                state.set(field, value);
            }
        }
    }
    if algorithm == OptimAlgorithm::TrustRegionReflective {
        for (field, value) in [
            (FunctionTolerance, Value::Num(1e-6)),
            (TolPCG, Value::Num(0.1)),
            (PrecondBandWidth, Value::Num(0.0)),
            (SubproblemAlgorithm, state::text(runtime, "cg")?),
            (
                MaxPCGIter,
                crate::builtins::math::optim::common::output_callback::empty(runtime)?,
            ),
            (
                HessianFcn,
                crate::builtins::math::optim::common::output_callback::empty(runtime)?,
            ),
            (
                HessianMultiplyFcn,
                crate::builtins::math::optim::common::output_callback::empty(runtime)?,
            ),
            (
                HessPattern,
                crate::builtins::math::optim::common::output_callback::empty(runtime)?,
            ),
        ] {
            if state
                .get(field)
                .is_none_or(|value| matches!(value,Value::Tensor(t)if t.is_empty()))
            {
                state.set(field, value);
            }
        }
    }
    let criterion = if algorithm == OptimAlgorithm::TrustRegionReflective {
        FunctionTolerance
    } else {
        OptimalityTolerance
    };
    if let Some(legacy) = state
        .get(TolFun)
        .filter(|value| !matches!(value,Value::Tensor(t)if t.is_empty()))
    {
        let origin = state.origin(TolFun).expect("stored legacy binding");
        let modern = state.origin(criterion);
        let prefer_modern =
            modern == Some(OptionOrigin::Explicit) && origin == OptionOrigin::Imported;
        if !prefer_modern {
            if let Some(value) = state.get(criterion) {
                if modern != Some(OptionOrigin::Default)
                    && modern == Some(origin)
                    && value != legacy
                {
                    return Err(optimoptions_error_with(
                        &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                        format!("optimoptions: conflicting TolFun and {}", criterion.name()),
                    ));
                }
            }
            let value = legacy.clone();
            state.set_with_origin(criterion, value, origin);
        }
    }
    state.remove(TolFun);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn numeric_empty_resets_known_defaults_under_actual_selected_algorithm() {
        use OptimOptionField::*;
        let mut state = default_options(None, Solver::Fmincon).unwrap();
        state.set_with_origin(
            Algorithm,
            crate::builtins::math::optim::common::output_callback::empty(None).unwrap(),
            state::OptionOrigin::Explicit,
        );
        normalize(None, &mut state).unwrap();
        assert_eq!(
            algorithm_value(Solver::Fmincon, state.get(Algorithm).unwrap()).unwrap(),
            OptimAlgorithm::InteriorPoint
        );
        state.set_with_origin(Algorithm, Value::from("sqp"), state::OptionOrigin::Explicit);
        state.set_with_origin(
            MaxIter,
            crate::builtins::math::optim::common::output_callback::empty(None).unwrap(),
            state::OptionOrigin::Explicit,
        );
        normalize(None, &mut state).unwrap();
        assert_eq!(state.get(MaxIter), Some(&Value::Num(400.0)));
        assert!(matches!(state.get(MaxFunEvals),Some(Value::Tensor(value))if value.is_empty()));
    }

    #[test]
    fn reflective_defaults_and_legacy_function_criterion_are_independent() {
        use OptimOptionField::*;
        let mut state = default_options(None, Solver::Fmincon).unwrap();
        state.set_with_origin(
            Algorithm,
            Value::from("trust-region-reflective"),
            state::OptionOrigin::Explicit,
        );
        state.set_with_origin(TolFun, Value::Num(2e-5), state::OptionOrigin::Explicit);
        state.set_with_origin(
            OptimalityTolerance,
            Value::Num(3e-7),
            state::OptionOrigin::Explicit,
        );
        normalize(None, &mut state).unwrap();
        assert_eq!(state.get(FunctionTolerance), Some(&Value::Num(2e-5)));
        assert_eq!(state.get(OptimalityTolerance), Some(&Value::Num(3e-7)));
        assert_eq!(state.get(TolPCG), Some(&Value::Num(0.1)));
        assert_eq!(state.get(PrecondBandWidth), Some(&Value::Num(0.0)));
        assert_eq!(state.get(MaxIter), Some(&Value::Num(400.0)));
        assert!(matches!(state.get(MaxPCGIter),Some(Value::Tensor(value))if value.is_empty()));
        assert!(state.get(TolFun).is_none());
    }
}
