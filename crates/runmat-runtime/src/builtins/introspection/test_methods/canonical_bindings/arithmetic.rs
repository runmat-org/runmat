use super::super::*;
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationPolicy, HostAllocationReceipt,
};

#[runmat_macros::runtime_builtin(
    name = "OverIdx.plus",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::arithmetic"
)]
pub(crate) async fn overidx_plus_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::OVERIDX_PLUS_DESCRIPTOR,
        &runmat_builtins::OVERIDX_FIXTURE_ERROR_OUTPUTS,
        "OverIdx.plus",
    )?;
    if args.len() != 2 {
        return Err(crate::runtime_descriptor_error(
            "OverIdx.plus",
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
    let result = overidx_plus(obj, rhs).await;
    drop(args);
    result
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.times",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::arithmetic"
)]
pub(crate) async fn overidx_times_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::OVERIDX_TIMES_DESCRIPTOR,
        &runmat_builtins::OVERIDX_FIXTURE_ERROR_OUTPUTS,
        "OverIdx.times",
    )?;
    if args.len() != 2 {
        return Err(crate::runtime_descriptor_error(
            "OverIdx.times",
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
    let result = overidx_times(obj, rhs).await;
    drop(args);
    result
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.mtimes",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::arithmetic"
)]
pub(crate) async fn overidx_mtimes_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::OVERIDX_MTIMES_DESCRIPTOR,
        &runmat_builtins::OVERIDX_FIXTURE_ERROR_OUTPUTS,
        "OverIdx.mtimes",
    )?;
    if args.len() != 2 {
        return Err(crate::runtime_descriptor_error(
            "OverIdx.mtimes",
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
    let result = overidx_mtimes(obj, rhs).await;
    drop(args);
    result
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.uplus",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::arithmetic"
)]
pub(crate) async fn overidx_uplus_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::OVERIDX_UPLUS_DESCRIPTOR,
        &runmat_builtins::OVERIDX_FIXTURE_ERROR_OUTPUTS,
        "OverIdx.uplus",
    )?;
    if args.len() != 1 {
        return Err(crate::runtime_descriptor_error(
            "OverIdx.uplus",
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
    let result = overidx_uplus(obj).await;
    drop(args);
    result
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.rdivide",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::arithmetic"
)]
pub(crate) async fn overidx_rdivide_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::OVERIDX_RDIVIDE_DESCRIPTOR,
        &runmat_builtins::OVERIDX_FIXTURE_ERROR_OUTPUTS,
        "OverIdx.rdivide",
    )?;
    if args.len() != 2 {
        return Err(crate::runtime_descriptor_error(
            "OverIdx.rdivide",
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
    let result = overidx_rdivide(obj, rhs).await;
    drop(args);
    result
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.mrdivide",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::arithmetic"
)]
pub(crate) async fn overidx_mrdivide_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::OVERIDX_MRDIVIDE_DESCRIPTOR,
        &runmat_builtins::OVERIDX_FIXTURE_ERROR_OUTPUTS,
        "OverIdx.mrdivide",
    )?;
    if args.len() != 2 {
        return Err(crate::runtime_descriptor_error(
            "OverIdx.mrdivide",
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
    let result = overidx_mrdivide(obj, rhs).await;
    drop(args);
    result
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.ldivide",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::arithmetic"
)]
pub(crate) async fn overidx_ldivide_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::OVERIDX_LDIVIDE_DESCRIPTOR,
        &runmat_builtins::OVERIDX_FIXTURE_ERROR_OUTPUTS,
        "OverIdx.ldivide",
    )?;
    if args.len() != 2 {
        return Err(crate::runtime_descriptor_error(
            "OverIdx.ldivide",
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
    let result = overidx_ldivide(obj, rhs).await;
    drop(args);
    result
}

#[runmat_macros::runtime_builtin(
    name = "OverIdx.mldivide",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::arithmetic"
)]
pub(crate) async fn overidx_mldivide_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::OVERIDX_MLDIVIDE_DESCRIPTOR,
        &runmat_builtins::OVERIDX_FIXTURE_ERROR_OUTPUTS,
        "OverIdx.mldivide",
    )?;
    if args.len() != 2 {
        return Err(crate::runtime_descriptor_error(
            "OverIdx.mldivide",
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
    let result = overidx_mldivide(obj, rhs).await;
    drop(args);
    result
}
