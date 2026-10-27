//! Canonical expression-function bindings.
use super::super::*;
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.sqrt",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::elementary"
)]
async fn sqrt(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_SQRT_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "sqrt" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.exp",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::elementary"
)]
async fn exp(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_EXP_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "exp" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.log",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::elementary"
)]
async fn log(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_LOG_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "log" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.sin",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::elementary"
)]
async fn sin(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_SIN_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "sin" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.cos",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::elementary"
)]
async fn cos(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_COS_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "cos" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.tan",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::elementary"
)]
async fn tan(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_TAN_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "tan" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.sec",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::elementary"
)]
async fn sec(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_SEC_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "sec" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.csc",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::elementary"
)]
async fn csc(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_CSC_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "csc" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.cot",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::elementary"
)]
async fn cot(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_COT_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "cot" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.asin",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::elementary"
)]
async fn asin(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_ASIN_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "asin" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.acos",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::elementary"
)]
async fn acos(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_ACOS_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "acos" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.atan",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::elementary"
)]
async fn atan(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_ATAN_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "atan" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.asec",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::elementary"
)]
async fn asec(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_ASEC_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "asec" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.acsc",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::elementary"
)]
async fn acsc(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_ACSC_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "acsc" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.acot",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::functions::elementary"
)]
async fn acot(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_ACOT_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::super::functions::builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "acot" },
        1,
        super::super::super::functions::FunctionRole::Ordinary,
        args.view(),
        policy,
        &mut cooperation,
    )
    .await
}
