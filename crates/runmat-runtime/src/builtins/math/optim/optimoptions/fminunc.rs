//! Final algorithm-aware normalization. Origins exist only during this call;
//! imported nonempty values are never guessed to be unset from their payload.
use super::*;
use state::OptionOrigin;

fn nonempty(value: &Value) -> bool {
    !matches!(value, Value::Tensor(t) if t.is_empty())
}

pub(super) fn normalize(
    runtime: Option<&crate::context::RuntimeContext>,
    state: &mut OptionState,
) -> BuiltinResult<()> {
    use OptimOptionField::*;
    let defaults = default_options(runtime, Solver::Fminunc)?;
    for (key, default) in defaults.fields() {
        if let OptionKey::Field(field) = key {
            if state.get(field).is_some_and(|value| !nonempty(value)) {
                state.set(field, default.clone());
            }
        }
    }
    let algorithm = state
        .get(Algorithm)
        .filter(|value| nonempty(value))
        .map(|value| algorithm_value(Solver::Fminunc, value))
        .transpose()?
        .unwrap_or(OptimAlgorithm::QuasiNewton);
    let gradient = state
        .get(SpecifyObjectiveGradient)
        .filter(|value| nonempty(value))
        .map(|value| logical_value(SpecifyObjectiveGradient.name(), value))
        .transpose()?
        .unwrap_or(false);
    let trust = algorithm == OptimAlgorithm::TrustRegion && gradient;
    if state.origin(MaxFunEvals) == Some(OptionOrigin::Default) && trust {
        state.set(
            MaxFunEvals,
            crate::builtins::math::optim::common::output_callback::empty(runtime)?,
        );
    }
    if let Some(legacy) = state.get(TolFun).filter(|value| nonempty(value)) {
        let target = if trust {
            FunctionTolerance
        } else {
            OptimalityTolerance
        };
        let legacy_origin = state.origin(TolFun).expect("stored legacy origin");
        let modern_origin = state.origin(target);
        let prefer_modern = modern_origin == Some(OptionOrigin::Explicit)
            && legacy_origin == OptionOrigin::Imported;
        if !prefer_modern {
            if let Some(modern) = state.get(target).filter(|value| nonempty(value)) {
                if modern_origin != Some(OptionOrigin::Default)
                    && modern_origin == Some(legacy_origin)
                    && modern != legacy
                {
                    return Err(optimoptions_error_with(
                        &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                        format!("optimoptions: conflicting TolFun and {}", target.name()),
                    ));
                }
            }
            let value = legacy.clone();
            state.set_with_origin(target, value, legacy_origin);
        }
    }
    state.remove(TolFun);
    // Recognized inactive controls remain inspectable, but the runtime planner
    // does not gather, validate or invoke them under the effective QN algorithm.
    // Empty controls retain their explicit unset marker. Dimension-dependent
    // limits resolve only in the solver, never by an
    // expression evaluator or hidden persisted provenance record.
    Ok(())
}
