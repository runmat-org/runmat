pub use runmat_builtins::REGISTER_TEST_CLASSES_DESCRIPTOR;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

pub async fn register_test_classes_builtin() -> crate::BuiltinResult<Value> {
    crate::register_test_classes_builtin().await
}

#[runtime_builtin(
    name = "__register_test_classes",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_classes"
)]
pub async fn register_test_classes_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &REGISTER_TEST_CLASSES_DESCRIPTOR,
        &runmat_builtins::REGISTER_TEST_CLASSES_ERROR_OUTPUTS,
        "__register_test_classes",
    )?;
    if !args.is_empty() {
        return Err(crate::runtime_descriptor_error(
            "__register_test_classes",
            &runmat_builtins::REGISTER_TEST_CLASSES_ERROR_ARITY,
        ));
    }
    use crate::context::host_allocation::{
        host_allocation_runtime_error, HostAllocationPolicy, HostAllocationReceipt,
    };
    let policy = HostAllocationReceipt::policy(runtime.as_ref()).map_err(|cause| {
        host_allocation_runtime_error(
            runtime.as_ref(),
            "RunMat:Placement:ResourceLimit",
            "__register_test_classes: allocation authority unavailable",
            cause,
        )
    })?;
    if policy == HostAllocationPolicy::Managed {
        return Err(crate::build_runtime_error(
            "__register_test_classes: construction backing has no certified finite-resource bound",
        )
        .with_builtin("__register_test_classes")
        .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
        .build());
    }

    let future = async {
        let result = register_test_classes_builtin().await;
        drop(args);
        result
    };
    match runtime.as_ref() {
        Some(context) => context.scope(future).await,
        None => future.await,
    }
}
