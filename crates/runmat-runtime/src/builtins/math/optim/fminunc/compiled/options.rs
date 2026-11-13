//! The actual fminunc option owner rejects unsupported model derivatives first.
use super::*;
pub(in crate::builtins::math::optim) async fn prepare(
    source: Option<&Value>,
    variables: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<PreparedOptions> {
    crate::context::host_allocation::require_unbounded_workspace(
        cooperation.runtime(),
        NAME,
        "modeled unconstrained optimization requires unbounded compiler and numerical workspace",
    )?;
    if source.is_some_and(|value| {
        !matches!(value, Value::Struct(_))
            && !matches!(value, Value::Tensor(array) if array.is_empty())
    }) {
        return Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_ARGUMENT,
            "options must be a scalar structure or []",
        ));
    }
    let source = parse_options(source)?;
    let runtime = cooperation.runtime().cloned();
    let values = common::prepare_modeled_fminunc_options(
        source,
        Some(&FMINUNC_OPTION_EXTENSION),
        Some(&FMINUNC_RESIDENT_EXTENSION),
        runtime.as_ref(),
        cooperation,
    )
    .await?;
    let controls =
        FminuncOptions::from_struct(values.as_ref().map(|value| value.structure()), variables)?;
    if controls.parallel != crate::execution::CallbackParallelMode::Off {
        return Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_ARGUMENT,
            "modeled parallel derivatives require an admitted portable model source",
        ));
    }
    if controls.trust_region_fallback {
        reporting::trust_region_fallback(cooperation.runtime())?;
    }
    Ok(PreparedOptions { controls, values })
}
