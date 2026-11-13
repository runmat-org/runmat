//! Original ordered instance/static/class fallback after a registry miss.
use super::*;
pub(super) async fn execute(
    request: Request<'_>,
) -> Result<Option<RootedValueSequence>, RuntimeError> {
    if request.policy == RetainedDispatchPolicy::RegisteredOnly {
        return Ok(None);
    }
    if let Some(result) = try_call_retained_instance_method_sequence(
        request.name,
        &request.args,
        request.output_count,
        request.context.as_ref(),
    )
    .await?
    {
        return compatibility_checked_builtin_sequence(request.name, request.args.view(), result)
            .map(Some);
    }
    if let Some(result) = try_call_registered_static_method(
        request.name,
        &request.args,
        request.output_count,
        request.context.as_ref(),
    )
    .await?
    {
        return Ok(Some(result));
    }
    if crate::class_registry::registered_constructor_identity(request.name).is_some() {
        let result = call_retained_class_constructor(
            request.name,
            request.args.fork(),
            request.output_count,
            request.context.as_ref(),
        )
        .await?;
        let sequence = result.try_map_owned(
            crate::call::arguments::adapt_legacy_builtin_result,
            |error| error.into_runtime_error(),
        )?;
        return compatibility_checked_builtin_sequence(request.name, request.args.view(), sequence)
            .map(Some);
    }
    Ok(None)
}
