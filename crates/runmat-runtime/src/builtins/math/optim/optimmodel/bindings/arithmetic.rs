//! Typed canonical expression operators.
use super::*;
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.uplus",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::arithmetic"
)]
async fn uplus(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_UPLUS_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    if args.len() != 1 {
        return Err(policy.invalid("invalid operator arity"));
    }
    super::super::operators::unary(
        runmat_types::OperatorKind::UnaryPlus,
        &args[0],
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.uminus",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::arithmetic"
)]
async fn uminus(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_UMINUS_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    if args.len() != 1 {
        return Err(policy.invalid("invalid operator arity"));
    }
    super::super::operators::unary(
        runmat_types::OperatorKind::UnaryMinus,
        &args[0],
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.transpose",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::arithmetic"
)]
async fn transpose(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_TRANSPOSE_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    if args.len() != 1 {
        return Err(policy.invalid("invalid operator arity"));
    }
    super::super::operators::unary(
        runmat_types::OperatorKind::Transpose,
        &args[0],
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.ctranspose",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::arithmetic"
)]
async fn ctranspose(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_CTRANSPOSE_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    if args.len() != 1 {
        return Err(policy.invalid("invalid operator arity"));
    }
    super::super::operators::unary(
        runmat_types::OperatorKind::ConjugateTranspose,
        &args[0],
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.plus",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::arithmetic"
)]
async fn plus(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_PLUS_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    if args.len() != 2 {
        return Err(policy.invalid("invalid operator arity"));
    }
    super::super::operators::binary(
        runmat_types::OperatorKind::Add,
        &args[0],
        &args[1],
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.minus",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::arithmetic"
)]
async fn minus(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_MINUS_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    if args.len() != 2 {
        return Err(policy.invalid("invalid operator arity"));
    }
    super::super::operators::binary(
        runmat_types::OperatorKind::Subtract,
        &args[0],
        &args[1],
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.mtimes",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::arithmetic"
)]
async fn mtimes(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_MTIMES_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    if args.len() != 2 {
        return Err(policy.invalid("invalid operator arity"));
    }
    super::super::operators::binary(
        runmat_types::OperatorKind::MatrixMultiply,
        &args[0],
        &args[1],
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.times",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::arithmetic"
)]
async fn times(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_TIMES_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    if args.len() != 2 {
        return Err(policy.invalid("invalid operator arity"));
    }
    super::super::operators::binary(
        runmat_types::OperatorKind::ElementwiseMultiply,
        &args[0],
        &args[1],
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.mpower",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::arithmetic"
)]
async fn mpower(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_MPOWER_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    if args.len() != 2 {
        return Err(policy.invalid("invalid operator arity"));
    }
    super::super::operators::binary(
        runmat_types::OperatorKind::MatrixPower,
        &args[0],
        &args[1],
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.power",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::arithmetic"
)]
async fn power(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_POWER_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    if args.len() != 2 {
        return Err(policy.invalid("invalid operator arity"));
    }
    super::super::operators::binary(
        runmat_types::OperatorKind::ElementwisePower,
        &args[0],
        &args[1],
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.mldivide",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::arithmetic"
)]
async fn mldivide(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_MLDIVIDE_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    if args.len() != 2 {
        return Err(policy.invalid("invalid operator arity"));
    }
    super::super::operators::binary(
        runmat_types::OperatorKind::Mldivide,
        &args[0],
        &args[1],
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.mrdivide",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::arithmetic"
)]
async fn mrdivide(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_MRDIVIDE_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    if args.len() != 2 {
        return Err(policy.invalid("invalid operator arity"));
    }
    super::super::operators::binary(
        runmat_types::OperatorKind::Mrdivide,
        &args[0],
        &args[1],
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.rdivide",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::arithmetic"
)]
async fn rdivide(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_RDIVIDE_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    if args.len() != 2 {
        return Err(policy.invalid("invalid operator arity"));
    }
    super::super::operators::binary(
        runmat_types::OperatorKind::ElementwiseDivide,
        &args[0],
        &args[1],
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.ldivide",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::arithmetic"
)]
async fn ldivide(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_LDIVIDE_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    if args.len() != 2 {
        return Err(policy.invalid("invalid operator arity"));
    }
    super::super::operators::binary(
        runmat_types::OperatorKind::ElementwiseLeftDivide,
        &args[0],
        &args[1],
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.le",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::arithmetic"
)]
async fn le(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_LE_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    if args.len() != 2 {
        return Err(policy.invalid("invalid operator arity"));
    }
    super::super::operators::binary(
        runmat_types::OperatorKind::LessEqual,
        &args[0],
        &args[1],
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.ge",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::arithmetic"
)]
async fn ge(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_GE_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    if args.len() != 2 {
        return Err(policy.invalid("invalid operator arity"));
    }
    super::super::operators::binary(
        runmat_types::OperatorKind::GreaterEqual,
        &args[0],
        &args[1],
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.eq",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::arithmetic"
)]
async fn eq(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_EQ_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    if args.len() != 2 {
        return Err(policy.invalid("invalid operator arity"));
    }
    super::super::operators::binary(
        runmat_types::OperatorKind::Equal,
        &args[0],
        &args[1],
        policy,
        &mut cooperation,
    )
    .await
}
