//! Preparation of an admitted dense floating pair. Controls stay with the caller.
use crate::builtins::common::broadcast::BroadcastPlan;
use crate::context::preparation::{
    execute_request_host_numerical, AdmittedBuiltinPreparation, BuiltinPreparationAdmission,
};
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease};
use crate::BuiltinResult;
use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogIdentity, BuiltinPreparationFormId};
use runmat_value::Value;
use std::future::Future;
pub(crate) mod managed;

pub(super) async fn execute<F, Fut>(
    context: Option<&RuntimeContext>,
    identity: BuiltinCatalogIdentity,
    operator: runmat_types::OperatorKind,
    input_count: usize,
    left: Value,
    right: Value,
    body: F,
) -> BuiltinResult<RootedValue>
where
    F: FnOnce(Value, Value) -> Fut,
    Fut: Future<Output = BuiltinResult<RootedValue>>,
{
    execute_in_context(context, identity, operator, input_count, left, right, body).await
}

/// Explicit context-bearing version for canonical bindings. The preparation
/// service receives the same instruction-scoped context as the selected body.
pub(super) async fn execute_in_context<F, Fut>(
    context: Option<&RuntimeContext>,
    identity: BuiltinCatalogIdentity,
    operator: runmat_types::OperatorKind,
    input_count: usize,
    left: Value,
    right: Value,
    body: F,
) -> BuiltinResult<RootedValue>
where
    F: FnOnce(Value, Value) -> Fut,
    Fut: Future<Output = BuiltinResult<RootedValue>>,
{
    let _inputs = RootedValueLease::for_values([&left, &right].into_iter())
        .map_err(|cause| cause.into_runtime_error())?;
    if input_count == 2 {
        if let Some(route) = crate::builtins::math::host_route::canonical(
            context,
            crate::context::VmFusionHostInstruction::ElementwiseBinary,
            [&left, &right].into_iter(),
        ) {
            let numerical_output = route.take_numerical_output(
                context.unwrap(),
                crate::context::VmFusionHostInstruction::ElementwiseBinary,
                BuiltinBindingIdentity {
                    builtin: identity,
                    variant: "default",
                },
            )?;
            let output = match numerical_output {
                Some(output) => {
                    managed::execute(context.unwrap(), output, operator, &left, &right)?
                }
                None => body(left, right).await?,
            };
            route.confirm_host_branch(
                context.unwrap(),
                crate::context::VmFusionHostInstruction::ElementwiseBinary,
            );
            return Ok(output);
        }
    }
    let admitted = input_count == 2
        && crate::builtins::math::host_route::real_floating_shape(&left)
            .zip(crate::builtins::math::host_route::real_floating_shape(
                &right,
            ))
            .is_some_and(|(left, right)| BroadcastPlan::new(left, right).is_ok());
    if !admitted {
        // The owning implementation supplies its established error and handles
        // the other representations. In particular, invalid shapes never upload.
        return body(left, right).await;
    }
    let admission = BuiltinPreparationAdmission {
        binding: BuiltinBindingIdentity {
            builtin: identity,
            variant: "default",
        },
        form: BuiltinPreparationFormId(0),
        signature: 0,
        input_count,
        requested_outputs: crate::output_count::current_output_count(),
    };
    let execute = |mut values: Vec<Value>, numerical_output| {
        let right = values.remove(1);
        let left = values.remove(0);
        let route = crate::builtins::math::host_route::canonical(
            context,
            crate::context::VmFusionHostInstruction::ElementwiseBinary,
            [&left, &right].into_iter(),
        );
        async move {
            let output = match numerical_output {
                Some(output) => managed::execute(
                    context.ok_or_else(managed::invalid)?,
                    output,
                    operator,
                    &left,
                    &right,
                )?,
                None => {
                    if let (Some(_), Some(context)) = (&route, context) {
                        managed::require_unbounded_host_output(context)?;
                    }
                    body(left, right).await?
                }
            };
            if let Some(route) = route {
                route.confirm_host_branch(
                    context.unwrap(),
                    crate::context::VmFusionHostInstruction::ElementwiseBinary,
                );
            }
            Ok(output)
        }
    };
    let request = AdmittedBuiltinPreparation::new(admission, vec![left, right])?;
    execute_request_host_numerical(context, request, execute).await
}
