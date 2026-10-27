//! Execute the retained registration after its normal instance override.
use super::*;
pub(super) async fn execute(
    mut request: Request<'_>,
    selected: selection::SelectedRegisteredBuiltin,
) -> Result<Option<RootedValueSequence>, RuntimeError> {
    let instance = if request.policy == RetainedDispatchPolicy::Normal {
        try_call_retained_instance_method_sequence(
            request.name,
            &request.args,
            request.output_count,
            request.context.as_ref(),
        )
        .await?
    } else {
        None
    };
    if let Some(result) = instance {
        return compatibility_checked_builtin_sequence(request.name, request.args.view(), result)
            .map(Some);
    }
    let selection::SelectedRegisteredBuiltin {
        origin,
        implementation,
        authority,
    } = selected;
    let prepared = preparation::prepare_retained_selected(
        origin,
        request.name,
        request.args.fork(),
        request.context.as_ref(),
    )
    .await?;
    let result = selected_execution::execute_retained(
        implementation,
        prepared,
        request.context.take(),
        request.output_count,
        request.observation.as_deref(),
        authority,
    )
    .await?;
    compatibility_checked_builtin_sequence(request.name, request.args.view(), result).map(Some)
}
