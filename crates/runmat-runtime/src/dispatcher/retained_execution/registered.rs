//! Execute the retained registration after its normal instance override.
use super::*;
use futures::future::Either;

pub(super) fn execute(
    request: Request<'_>,
    selected: selection::SelectedRegisteredBuiltin,
) -> impl Future<Output = Result<Option<RootedValueSequence>, RuntimeError>> + '_ {
    crate::execution::after_result(
        async move {
            let instance = if request.policy == RetainedDispatchPolicy::Normal
                && !selected
                    .implementation
                    .is_handle_validity_query(request.args.first())
            {
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
            Ok((request, selected, instance))
        },
        |(request, selected, instance)| match instance {
            Some(result) => Either::Left(std::future::ready(
                compatibility_checked_builtin_sequence(request.name, request.args.view(), result)
                    .map(Some),
            )),
            None => Either::Right(registered(request, selected)),
        },
    )
}

#[inline(always)]
fn registered(
    request: Request<'_>,
    selected: selection::SelectedRegisteredBuiltin,
) -> impl Future<Output = Result<Option<RootedValueSequence>, RuntimeError>> + '_ {
    // Retire override and preparation frames before polling numerical work.
    // Request retains the original roots, output guard and observation owner
    // across both transitions; this adds no continuation allocation.
    crate::execution::after_result(
        async move {
            let prepared = preparation::prepare_retained_selected(
                selected.origin,
                request.name,
                request.args.fork(),
                request.context.as_ref(),
            )
            .await?;
            Ok((request, selected, prepared))
        },
        |(request, selected, prepared)| execute_prepared(request, selected, prepared),
    )
}

// Capture Request as a whole async-function argument. Field-wise capture in
// an async closure can retire its unused output guard before execution begins.
async fn execute_prepared(
    mut request: Request<'_>,
    selected: selection::SelectedRegisteredBuiltin,
    prepared: crate::call::BuiltinArguments,
) -> Result<Option<RootedValueSequence>, RuntimeError> {
    let result = selected_execution::execute_retained(
        selected.implementation,
        prepared,
        request.context.take(),
        request.output_count,
        request.observation.as_deref(),
        selected.authority,
    )
    .await?;
    compatibility_checked_builtin_sequence(request.name, request.args.view(), result).map(Some)
}
