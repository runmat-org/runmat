//! Construct only the selected branch and retain the original roots through
//! completion/adoption in one exactly admitted selected-body allocation.
use super::selection::{Branch, Selected};
use super::*;
use crate::builtins::math::host_route::{self, NumericHostRoute};
use crate::context::preparation::{
    execute_request_host_numerical, AdmittedBuiltinPreparation, PreparedHostNumericalOutput,
};
use crate::context::VmFusionHostInstruction;
use futures::future::Either;
use runmat_builtins::BuiltinBindingIdentity;

pub(super) fn execute<'a, F, Fut>(
    context: Option<&'a RuntimeContext>,
    identity: BuiltinCatalogIdentity,
    operator: runmat_types::OperatorKind,
    selected: Selected,
    body: F,
) -> crate::execution::RuntimeContinuationFuture<'a, RootedValue>
where
    F: FnOnce(Value, Value) -> Fut + 'a,
    Fut: Future<Output = BuiltinResult<RootedValue>> + 'a,
{
    let Selected {
        left,
        right,
        branch,
        inputs,
    } = selected;
    match branch {
        Branch::Ordinary => admit_selected(context, inputs, || body(left, right)),
        Branch::Host(route) => admit_selected(context, inputs, || {
            host(
                context.expect("selected host route has its original runtime"),
                identity,
                operator,
                left,
                right,
                route,
                body,
            )
        }),
        Branch::Admitted(admission) => admit_selected(context, inputs, || {
            admitted(context, admission, operator, left, right, body)
        }),
    }
}

// The preparation successor carries this small erased owner, not an enum of
// all three concrete bodies. Construct only the selected body at the existing
// layout-admission boundary, after its exact storage claim and before polling.
#[inline(never)]
fn admit_selected<'a, Fut>(
    context: Option<&RuntimeContext>,
    inputs: crate::execution::RootedValueLease,
    make: impl FnOnce() -> Fut,
) -> crate::execution::RuntimeContinuationFuture<'a, RootedValue>
where
    Fut: Future<Output = BuiltinResult<RootedValue>> + 'a,
{
    crate::execution::rooted_continuation::admit_optional_hostcall_with_contextual_error(
        context,
        || crate::execution::with_retained_owner(make(), inputs),
    )
    .unwrap_or_else(crate::execution::RuntimeContinuationFuture::rejected)
}

#[cfg(test)]
#[path = "selected_execution/tests.rs"]
mod tests;
fn host<'a, F, Fut>(
    context: &'a RuntimeContext,
    identity: BuiltinCatalogIdentity,
    operator: runmat_types::OperatorKind,
    left: Value,
    right: Value,
    route: NumericHostRoute,
    body: F,
) -> impl Future<Output = BuiltinResult<RootedValue>> + 'a
where
    F: FnOnce(Value, Value) -> Fut + 'a,
    Fut: Future<Output = BuiltinResult<RootedValue>> + 'a,
{
    let numerical_output = route.take_numerical_output(
        context,
        VmFusionHostInstruction::ElementwiseBinary,
        BuiltinBindingIdentity {
            builtin: identity,
            variant: "default",
        },
    );
    let future = match numerical_output {
        Err(error) => Either::Left(std::future::ready(Err(error))),
        Ok(Some(output)) => Either::Left(std::future::ready(managed::execute(
            context, output, operator, &left, &right,
        ))),
        Ok(None) => match managed::require_unbounded_host_output(context) {
            Err(error) => Either::Left(std::future::ready(Err(error))),
            Ok(()) => Either::Right(body(left, right)),
        },
    };
    crate::execution::after_result(future, move |output| async move {
        route.confirm_host_branch(context, VmFusionHostInstruction::ElementwiseBinary);
        Ok(output)
    })
}
fn admitted<'a, F, Fut>(
    context: Option<&'a RuntimeContext>,
    admission: crate::context::preparation::BuiltinPreparationAdmission,
    operator: runmat_types::OperatorKind,
    left: Value,
    right: Value,
    body: F,
) -> impl Future<Output = BuiltinResult<RootedValue>> + 'a
where
    F: FnOnce(Value, Value) -> Fut + 'a,
    Fut: Future<Output = BuiltinResult<RootedValue>> + 'a,
{
    match AdmittedBuiltinPreparation::new(admission, vec![left, right]) {
        Err(error) => Either::Left(std::future::ready(Err(error))),
        Ok(request) => Either::Right(execute_request_host_numerical(
            context,
            request,
            move |mut values, numerical_output| {
                let right = values.remove(1);
                let left = values.remove(0);
                let route = host_route::canonical(
                    context,
                    VmFusionHostInstruction::ElementwiseBinary,
                    [&left, &right].into_iter(),
                );
                prepared_body(
                    context,
                    operator,
                    left,
                    right,
                    numerical_output,
                    route,
                    body,
                )
            },
        )),
    }
}
fn prepared_body<'a, F, Fut>(
    context: Option<&'a RuntimeContext>,
    operator: runmat_types::OperatorKind,
    left: Value,
    right: Value,
    numerical_output: Option<PreparedHostNumericalOutput>,
    route: Option<NumericHostRoute>,
    body: F,
) -> impl Future<Output = BuiltinResult<RootedValue>> + 'a
where
    F: FnOnce(Value, Value) -> Fut + 'a,
    Fut: Future<Output = BuiltinResult<RootedValue>> + 'a,
{
    let future = match numerical_output {
        Some(output) => Either::Left(std::future::ready(
            context
                .ok_or_else(managed::invalid)
                .and_then(|context| managed::execute(context, output, operator, &left, &right)),
        )),
        None => {
            // A resident body uses its provider's allocation authority. Its
            // canonical host fallback separately refuses unprepared finite
            // allocation before any gather; do not demand a host output owner
            // before giving an admitted device operation a chance to execute.
            let resident =
                matches!(left, Value::GpuTensor(_)) || matches!(right, Value::GpuTensor(_));
            let permitted = if resident {
                Ok(())
            } else {
                context.map_or(Ok(()), managed::require_unbounded_host_output)
            };
            match permitted {
                Err(error) => Either::Left(std::future::ready(Err(error))),
                Ok(()) => Either::Right(body(left, right)),
            }
        }
    };
    crate::execution::after_result(future, move |output| async move {
        if let Some(route) = route {
            route.confirm_host_branch(
                context.expect("selected route has original context"),
                VmFusionHostInstruction::ElementwiseBinary,
            );
        }
        Ok(output)
    })
}
