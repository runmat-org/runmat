use super::super::*;
use crate::context::host_allocation::{host_allocation_runtime_error, HostAllocationReceipt};

#[runmat_macros::runtime_builtin(
    name = "OverIdx.lt",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::predicates"
)]
pub(crate) async fn overidx_lt_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::OVERIDX_LT_DESCRIPTOR,
        &runmat_builtins::OVERIDX_FIXTURE_ERROR_OUTPUTS,
        "OverIdx.lt",
    )?;
    if args.len() != 2 {
        return Err(crate::runtime_descriptor_error(
            "OverIdx.lt",
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
    let _ = policy;
    let obj: Value = args[0].clone();
    let rhs: Value = args[1].clone();
    let result = overidx_lt(obj, rhs).await;
    drop(args);
    result
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.gt",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::predicates"
)]
pub(crate) async fn overidx_gt_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::OVERIDX_GT_DESCRIPTOR,
        &runmat_builtins::OVERIDX_FIXTURE_ERROR_OUTPUTS,
        "OverIdx.gt",
    )?;
    if args.len() != 2 {
        return Err(crate::runtime_descriptor_error(
            "OverIdx.gt",
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
    let _ = policy;
    let obj: Value = args[0].clone();
    let rhs: Value = args[1].clone();
    let result = overidx_gt(obj, rhs).await;
    drop(args);
    result
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.eq",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::predicates"
)]
pub(crate) async fn overidx_eq_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::OVERIDX_EQ_DESCRIPTOR,
        &runmat_builtins::OVERIDX_FIXTURE_ERROR_OUTPUTS,
        "OverIdx.eq",
    )?;
    if args.len() != 2 {
        return Err(crate::runtime_descriptor_error(
            "OverIdx.eq",
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
    let _ = policy;
    let obj: Value = args[0].clone();
    let rhs: Value = args[1].clone();
    let result = overidx_eq(obj, rhs).await;
    drop(args);
    result
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.and",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::predicates"
)]
pub(crate) async fn overidx_and_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::OVERIDX_AND_DESCRIPTOR,
        &runmat_builtins::OVERIDX_FIXTURE_ERROR_OUTPUTS,
        "OverIdx.and",
    )?;
    if args.len() != 2 {
        return Err(crate::runtime_descriptor_error(
            "OverIdx.and",
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
    let _ = policy;
    let obj: Value = args[0].clone();
    let rhs: Value = args[1].clone();
    let result = overidx_and(obj, rhs).await;
    drop(args);
    result
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.or",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::predicates"
)]
pub(crate) async fn overidx_or_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::OVERIDX_OR_DESCRIPTOR,
        &runmat_builtins::OVERIDX_FIXTURE_ERROR_OUTPUTS,
        "OverIdx.or",
    )?;
    if args.len() != 2 {
        return Err(crate::runtime_descriptor_error(
            "OverIdx.or",
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
    let _ = policy;
    let obj: Value = args[0].clone();
    let rhs: Value = args[1].clone();
    let result = overidx_or(obj, rhs).await;
    drop(args);
    result
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.xor",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::predicates"
)]
pub(crate) async fn overidx_xor_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::OVERIDX_XOR_DESCRIPTOR,
        &runmat_builtins::OVERIDX_FIXTURE_ERROR_OUTPUTS,
        "OverIdx.xor",
    )?;
    if args.len() != 2 {
        return Err(crate::runtime_descriptor_error(
            "OverIdx.xor",
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
    let _ = policy;
    let obj: Value = args[0].clone();
    let rhs: Value = args[1].clone();
    let result = overidx_xor(obj, rhs).await;
    drop(args);
    result
}
