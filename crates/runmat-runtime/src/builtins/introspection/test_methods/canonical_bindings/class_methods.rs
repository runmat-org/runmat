use super::super::*;
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationPolicy, HostAllocationReceipt,
};

#[runmat_macros::runtime_builtin(
    name = "Point.move",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::class_methods"
)]
pub(crate) async fn point_move_method_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::POINT_MOVE_DESCRIPTOR,
        &runmat_builtins::CLASS_FIXTURE_ERROR_OUTPUTS,
        "Point.move",
    )?;
    if args.len() != 3 {
        return Err(crate::runtime_descriptor_error(
            "Point.move",
            &runmat_builtins::CLASS_FIXTURE_ERROR_ARITY,
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
    let dx: f64 = (&args[1]).try_into()?;
    let dy: f64 = (&args[2]).try_into()?;
    let result = point_move_method(obj, dx, dy).await;
    drop(args);
    result
}

#[runmat_macros::runtime_builtin(
    name = "Point.origin",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::class_methods"
)]
pub(crate) async fn point_origin_method_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::POINT_ORIGIN_DESCRIPTOR,
        &runmat_builtins::CLASS_FIXTURE_ERROR_OUTPUTS,
        "Point.origin",
    )?;
    if !args.is_empty() {
        return Err(crate::runtime_descriptor_error(
            "Point.origin",
            &runmat_builtins::CLASS_FIXTURE_ERROR_ARITY,
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

    let result = point_origin_method().await;
    drop(args);
    result
}

#[runmat_macros::runtime_builtin(
    name = "Shape.area",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::class_methods"
)]
pub(crate) async fn shape_area_method_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::SHAPE_AREA_DESCRIPTOR,
        &runmat_builtins::CLASS_FIXTURE_ERROR_OUTPUTS,
        "Shape.area",
    )?;
    if args.len() != 1 {
        return Err(crate::runtime_descriptor_error(
            "Shape.area",
            &runmat_builtins::CLASS_FIXTURE_ERROR_ARITY,
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
    let _obj: Value = args[0].clone();
    let result = shape_area_method(_obj).await;
    drop(args);
    result
}

#[runmat_macros::runtime_builtin(
    name = "Circle.area",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::class_methods"
)]
pub(crate) async fn circle_area_method_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::CIRCLE_AREA_DESCRIPTOR,
        &runmat_builtins::CLASS_FIXTURE_ERROR_OUTPUTS,
        "Circle.area",
    )?;
    if args.len() != 1 {
        return Err(crate::runtime_descriptor_error(
            "Circle.area",
            &runmat_builtins::CLASS_FIXTURE_ERROR_ARITY,
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
    let result = circle_area_method(obj).await;
    drop(args);
    result
}

#[runmat_macros::runtime_builtin(
    name = "Ctor.Ctor",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::class_methods"
)]
pub(crate) async fn ctor_ctor_method_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::CTOR_CTOR_DESCRIPTOR,
        &runmat_builtins::CLASS_FIXTURE_ERROR_OUTPUTS,
        "Ctor.Ctor",
    )?;
    if args.len() != 1 {
        return Err(crate::runtime_descriptor_error(
            "Ctor.Ctor",
            &runmat_builtins::CLASS_FIXTURE_ERROR_ARITY,
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
    let x: f64 = (&args[0]).try_into()?;
    let result = ctor_ctor_method(x).await;
    drop(args);
    result
}

#[runmat_macros::runtime_builtin(
    name = "PkgF.foo",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::class_methods"
)]
pub(crate) async fn pkgf_foo_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::PKGF_FOO_DESCRIPTOR,
        &runmat_builtins::CLASS_FIXTURE_ERROR_OUTPUTS,
        "PkgF.foo",
    )?;
    if !args.is_empty() {
        return Err(crate::runtime_descriptor_error(
            "PkgF.foo",
            &runmat_builtins::CLASS_FIXTURE_ERROR_ARITY,
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

    let result = pkgf_foo().await;
    drop(args);
    result
}

#[runmat_macros::runtime_builtin(
    name = "PkgG.foo",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::test_methods::canonical_bindings::class_methods"
)]
pub(crate) async fn pkgg_foo_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::PKGG_FOO_DESCRIPTOR,
        &runmat_builtins::CLASS_FIXTURE_ERROR_OUTPUTS,
        "PkgG.foo",
    )?;
    if !args.is_empty() {
        return Err(crate::runtime_descriptor_error(
            "PkgG.foo",
            &runmat_builtins::CLASS_FIXTURE_ERROR_ARITY,
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

    let result = pkgg_foo().await;
    drop(args);
    result
}
