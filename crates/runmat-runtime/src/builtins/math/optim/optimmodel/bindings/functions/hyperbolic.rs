//! Canonical expression-function bindings.
use super::super::*;
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.sinh",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::hyperbolic"
)]
async fn sinh(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_SINH_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "sinh" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.cosh",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::hyperbolic"
)]
async fn cosh(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_COSH_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "cosh" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.tanh",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::hyperbolic"
)]
async fn tanh(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_TANH_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "tanh" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.sech",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::hyperbolic"
)]
async fn sech(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_SECH_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "sech" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.csch",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::hyperbolic"
)]
async fn csch(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_CSCH_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "csch" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.coth",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::hyperbolic"
)]
async fn coth(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_COTH_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "coth" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.asinh",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::hyperbolic"
)]
async fn asinh(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_ASINH_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "asinh" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.acosh",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::hyperbolic"
)]
async fn acosh(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_ACOSH_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "acosh" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.atanh",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::hyperbolic"
)]
async fn atanh(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_ATANH_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "atanh" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.asech",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::hyperbolic"
)]
async fn asech(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_ASECH_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "asech" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.acsch",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::hyperbolic"
)]
async fn acsch(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_ACSCH_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "acsch" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.acoth",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::hyperbolic"
)]
async fn acoth(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_ACOTH_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "acoth" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
