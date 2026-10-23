use super::super::*;
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationPolicy, HostAllocationReceipt,
};

#[runmat_macros::runtime_builtin(
    name = "OverIdx.numArgumentsFromSubscript",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::indexing_serialization"
)]
pub(crate) async fn overidx_num_arguments_from_subscript_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::OVERIDX_NUM_ARGUMENTS_DESCRIPTOR,
        &runmat_builtins::OVERIDX_FIXTURE_ERROR_OUTPUTS,
        "OverIdx.numArgumentsFromSubscript",
    )?;
    if args.len() != 3 {
        return Err(crate::runtime_descriptor_error(
            "OverIdx.numArgumentsFromSubscript",
            &runmat_builtins::OVERIDX_FIXTURE_ERROR_ARITY,
        ));
    }
    let policy = HostAllocationReceipt::policy(runtime.as_ref()).map_err(|cause| {
        host_allocation_runtime_error(
            runtime.as_ref(),
            "RunMat:Placement:ResourceLimit",
            "fixture allocation authority unavailable",
            cause,
        )
    })?;
    if policy == HostAllocationPolicy::Managed {
        return Err(crate::build_runtime_error(
            "fixture object/alias backing has no certified finite-resource bound",
        )
        .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
        .build());
    }
    let obj: Value = args[0].clone();
    let _subscript: Value = args[1].clone();
    let _indexing_context: Value = args[2].clone();
    let result = overidx_num_arguments_from_subscript(obj, _subscript, _indexing_context).await;
    drop(args);
    result
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.saveobj",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::indexing_serialization"
)]
pub(crate) async fn overidx_saveobj_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::OVERIDX_SAVEOBJ_DESCRIPTOR,
        &runmat_builtins::OVERIDX_FIXTURE_ERROR_OUTPUTS,
        "OverIdx.saveobj",
    )?;
    if args.len() != 1 {
        return Err(crate::runtime_descriptor_error(
            "OverIdx.saveobj",
            &runmat_builtins::OVERIDX_FIXTURE_ERROR_ARITY,
        ));
    }
    let policy = HostAllocationReceipt::policy(runtime.as_ref()).map_err(|cause| {
        host_allocation_runtime_error(
            runtime.as_ref(),
            "RunMat:Placement:ResourceLimit",
            "fixture allocation authority unavailable",
            cause,
        )
    })?;
    if policy == HostAllocationPolicy::Managed {
        return Err(crate::build_runtime_error(
            "fixture object/alias backing has no certified finite-resource bound",
        )
        .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
        .build());
    }
    let obj: Value = args[0].clone();
    let result = overidx_saveobj(obj).await;
    drop(args);
    result
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.loadobj",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::indexing_serialization"
)]
pub(crate) async fn overidx_loadobj_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::OVERIDX_LOADOBJ_DESCRIPTOR,
        &runmat_builtins::OVERIDX_FIXTURE_ERROR_OUTPUTS,
        "OverIdx.loadobj",
    )?;
    if args.len() != 1 {
        return Err(crate::runtime_descriptor_error(
            "OverIdx.loadobj",
            &runmat_builtins::OVERIDX_FIXTURE_ERROR_ARITY,
        ));
    }
    let policy = HostAllocationReceipt::policy(runtime.as_ref()).map_err(|cause| {
        host_allocation_runtime_error(
            runtime.as_ref(),
            "RunMat:Placement:ResourceLimit",
            "fixture allocation authority unavailable",
            cause,
        )
    })?;
    if policy == HostAllocationPolicy::Managed {
        return Err(crate::build_runtime_error(
            "fixture object/alias backing has no certified finite-resource bound",
        )
        .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
        .build());
    }
    let payload: Value = args[0].clone();
    let result = overidx_loadobj(payload).await;
    drop(args);
    result
}
