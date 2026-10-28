//! Canonical expression-function bindings.
use super::super::*;
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.sum",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::reductions"
)]
async fn sum(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_SUM_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "sum" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.mean",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::reductions"
)]
async fn mean(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_MEAN_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "mean" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.prod",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::reductions"
)]
async fn prod(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_PROD_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "prod" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.trace",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::reductions"
)]
async fn trace(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_TRACE_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "trace" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.cumsum",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::reductions"
)]
async fn cumsum(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_CUMSUM_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "cumsum" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.cumprod",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::reductions"
)]
async fn cumprod(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_CUMPROD_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "cumprod" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.diff",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::reductions"
)]
async fn diff(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_DIFF_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "diff" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.reshape",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::reductions"
)]
async fn reshape(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_RESHAPE_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "reshape" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.diag",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::reductions"
)]
async fn diag(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_DIAG_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "diag" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.dot",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::reductions"
)]
async fn dot(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_DOT_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "dot" },
        2,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.norm",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::reductions"
)]
async fn norm(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_NORM_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "norm" },
        1,
        super::super::super::functions::FunctionRole::TwoNorm,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
