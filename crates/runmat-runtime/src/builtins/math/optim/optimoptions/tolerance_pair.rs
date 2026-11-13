//! Legacy TolFun supplies a pair of distinct function and optimality controls.
//! Merge precedence uses only transient constructor origin, never a persisted
//! hidden ledger or a guess that an imported value was a default.
use super::*;
pub(super) fn normalize(
    runtime: Option<&crate::context::RuntimeContext>,
    solver: Solver,
    options: &mut OptionState,
) -> BuiltinResult<()> {
    use state::OptionOrigin;
    use OptimOptionField::*;
    let defaults = default_options(runtime, solver)?;
    for (key, value) in defaults.fields() {
        if let OptionKey::Field(field) = key {
            if field != MaxFunEvals
                && options
                    .get(field)
                    .is_some_and(|value| matches!(value, Value::Tensor(value) if value.is_empty()))
            {
                options.set(field, value.clone());
            }
        }
    }
    if let Some(legacy) = options
        .get(TolFun)
        .filter(|value| !matches!(value, Value::Tensor(value) if value.is_empty()))
    {
        let legacy = legacy.clone();
        let origin = options.origin(TolFun).expect("stored legacy option");
        for field in [FunctionTolerance, OptimalityTolerance] {
            let modern_origin = options.origin(field);
            if modern_origin == Some(OptionOrigin::Explicit) && origin == OptionOrigin::Imported {
                continue;
            }
            if let Some(modern) = options.get(field) {
                if modern_origin == Some(origin)
                    && modern_origin != Some(OptionOrigin::Default)
                    && modern != &legacy
                {
                    return Err(optimoptions_error_with(
                        &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                        format!(
                            "optimoptions: conflicting TolFun and {} controls",
                            field.name()
                        ),
                    ));
                }
            }
            options.set_with_origin(field, legacy.clone(), origin);
        }
    }
    options.remove(TolFun);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transient_legacy_precedence_keeps_both_real_stopping_controls() {
        for solver in [Solver::Fminimax, Solver::Fgoalattain, Solver::Lsqnonlin] {
            let mut state = default_options(None, solver).unwrap();
            state.set_with_origin(
                OptimOptionField::TolFun,
                Value::Num(1e-4),
                state::OptionOrigin::Explicit,
            );
            normalize(None, solver, &mut state).unwrap();
            assert!(state.get(OptimOptionField::TolFun).is_none());
            for field in [
                OptimOptionField::FunctionTolerance,
                OptimOptionField::OptimalityTolerance,
            ] {
                assert_eq!(state.get(field), Some(&Value::Num(1e-4)));
            }
            state.set_with_origin(
                OptimOptionField::TolFun,
                Value::Num(1e-3),
                state::OptionOrigin::Imported,
            );
            state.set_with_origin(
                OptimOptionField::FunctionTolerance,
                Value::Num(1e-5),
                state::OptionOrigin::Explicit,
            );
            normalize(None, solver, &mut state).unwrap();
            assert_eq!(
                state.get(OptimOptionField::FunctionTolerance),
                Some(&Value::Num(1e-5))
            );
            state.set_with_origin(
                OptimOptionField::TolFun,
                Value::Num(1e-2),
                state::OptionOrigin::Explicit,
            );
            assert!(normalize(None, solver, &mut state).is_err());
        }
    }
}
