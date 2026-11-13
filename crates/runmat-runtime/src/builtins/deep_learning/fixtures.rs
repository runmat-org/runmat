//! Executable autodiff callback fixtures with canonical production bindings.
//! Catalog audience controls public discovery; binding availability is shared
//! between test and production builds.

use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValues};
use crate::BuiltinResult;
use runmat_builtins::{
    BuiltinCatalogIdentity, DLGRADIENT_CATALOG_ENTRY, FORWARD_CATALOG_ENTRY, SUM_CATALOG_ENTRY,
    TIMES_CATALOG_ENTRY,
};
use runmat_value::Value;

async fn call_one(
    runtime: Option<&RuntimeContext>,
    identity: BuiltinCatalogIdentity,
    args: &[Value],
) -> BuiltinResult<RootedValue> {
    match runtime {
        Some(runtime) => {
            crate::call_builtin_async_with_outputs_in_context(runtime, identity.name, args, 1).await
        }
        None => crate::call_builtin_async_with_outputs(identity.name, args, 1).await,
    }
}

#[runmat_macros::runtime_builtin(
    name = "__deep_learning_square_grad",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::deep_learning::fixtures"
)]
async fn deep_learning_square_grad_helper(
    runtime: Option<RuntimeContext>,
    x: Value,
) -> BuiltinResult<RootedValue> {
    let x = RootedValue::new(x).map_err(|error| error.into_runtime_error())?;
    x.with_owned_async(|x| async move {
        let y = call_one(
            runtime.as_ref(),
            TIMES_CATALOG_ENTRY.identity,
            &[x.clone(), x.clone()],
        )
        .await?;
        y.with_owned_async(|y| async move {
            let loss = call_one(
                runtime.as_ref(),
                SUM_CATALOG_ENTRY.identity,
                &[y, Value::String("all".into())],
            )
            .await?;
            loss.with_owned_async(|loss| async move {
                let gradient = call_one(
                    runtime.as_ref(),
                    DLGRADIENT_CATALOG_ENTRY.identity,
                    &[loss.clone(), x],
                )
                .await?;
                gradient.with_owned(|gradient| {
                    RootedValue::new(Value::output_list(vec![loss, gradient]))
                        .map_err(|error| error.into_runtime_error())
                })
            })
            .await
        })
        .await
    })
    .await
}

#[runmat_macros::runtime_builtin(
    name = "__deep_learning_network_grad",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::deep_learning::fixtures"
)]
async fn deep_learning_network_grad_helper(
    runtime: Option<RuntimeContext>,
    net: Value,
    x: Value,
) -> BuiltinResult<RootedValue> {
    let inputs = RootedValues::new(vec![net, x]).map_err(|error| error.into_runtime_error())?;
    inputs
        .with_unbounded_values_async(|mut inputs| async move {
            let x = inputs.pop().expect("fixture input x");
            let net = inputs.pop().expect("fixture input network");
            let y = call_one(
                runtime.as_ref(),
                FORWARD_CATALOG_ENTRY.identity,
                &[net.clone(), x],
            )
            .await?;
            y.with_owned_async(|y| async move {
                let loss = call_one(
                    runtime.as_ref(),
                    SUM_CATALOG_ENTRY.identity,
                    &[y, Value::String("all".into())],
                )
                .await?;
                loss.with_owned_async(|loss| async move {
                    call_one(
                        runtime.as_ref(),
                        DLGRADIENT_CATALOG_ENTRY.identity,
                        &[loss, net],
                    )
                    .await
                })
                .await
            })
            .await
        })
        .await?
}
