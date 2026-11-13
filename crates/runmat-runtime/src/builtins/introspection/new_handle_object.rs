pub use runmat_builtins::NEW_HANDLE_OBJECT_DESCRIPTOR;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

pub async fn new_handle_object_builtin(class_name: String) -> crate::BuiltinResult<Value> {
    crate::new_handle_object_builtin(class_name).await
}

#[runtime_builtin(
    name = "new_handle_object",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::new_handle_object"
)]
pub async fn new_handle_object_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &NEW_HANDLE_OBJECT_DESCRIPTOR,
        &runmat_builtins::NEW_HANDLE_OBJECT_ERROR_OUTPUTS,
        "new_handle_object",
    )?;
    if args.len() != 1 {
        return Err(crate::runtime_descriptor_error(
            "new_handle_object",
            &runmat_builtins::NEW_HANDLE_OBJECT_ERROR_ARITY,
        ));
    }
    use crate::context::host_allocation::{
        host_allocation_runtime_error, HostAllocationPolicy, HostAllocationReceipt,
    };
    let policy = HostAllocationReceipt::policy(runtime.as_ref()).map_err(|cause| {
        host_allocation_runtime_error(
            runtime.as_ref(),
            "RunMat:Placement:ResourceLimit",
            "new_handle_object: allocation authority unavailable",
            cause,
        )
    })?;
    if policy == HostAllocationPolicy::Managed {
        return Err(crate::build_runtime_error(
            "new_handle_object: construction backing has no certified finite-resource bound",
        )
        .with_builtin("new_handle_object")
        .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
        .build());
    }
    let future = async {
        let first: String = (&args[0]).try_into()?;

        let result = new_handle_object_builtin(first).await;
        drop(args);
        result
    };
    match runtime.as_ref() {
        Some(context) => context.scope(future).await,
        None => future.await,
    }
}
