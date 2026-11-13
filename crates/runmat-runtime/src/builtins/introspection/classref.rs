pub use runmat_builtins::CLASSREF_DESCRIPTOR;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

pub async fn classref_builtin(class_name: String) -> crate::BuiltinResult<Value> {
    let identity = runmat_types::ClassIdentity::new(&class_name)
        .map_err(|error| crate::build_runtime_error(error.to_string()).build())?;
    if identity.is(crate::builtins::introspection::object_indexing::INDEXING_CONTEXT_CLASS)
        || identity
            .is(crate::builtins::introspection::object_indexing::LEGACY_INDEXING_CONTEXT_CLASS)
    {
        crate::builtins::introspection::object_indexing::ensure_indexing_context_classes_registered(
        );
    }
    crate::classref_builtin(class_name).await
}

#[runtime_builtin(
    name = "classref",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::classref"
)]
pub async fn classref_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &CLASSREF_DESCRIPTOR,
        &runmat_builtins::CLASSREF_ERROR_OUTPUTS,
        "classref",
    )?;
    if args.len() != 1 {
        return Err(crate::runtime_descriptor_error(
            "classref",
            &runmat_builtins::CLASSREF_ERROR_ARITY,
        ));
    }
    use crate::context::host_allocation::{
        host_allocation_runtime_error, HostAllocationPolicy, HostAllocationReceipt,
    };
    let policy = HostAllocationReceipt::policy(runtime.as_ref()).map_err(|cause| {
        host_allocation_runtime_error(
            runtime.as_ref(),
            "RunMat:Placement:ResourceLimit",
            "classref: allocation authority unavailable",
            cause,
        )
    })?;
    if policy == HostAllocationPolicy::Managed {
        return Err(crate::build_runtime_error(
            "classref: construction backing has no certified finite-resource bound",
        )
        .with_builtin("classref")
        .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
        .build());
    }
    let future = async {
        let first: String = (&args[0]).try_into()?;

        let result = classref_builtin(first).await;
        drop(args);
        result
    };
    match runtime.as_ref() {
        Some(context) => context.scope(future).await,
        None => future.await,
    }
}
