//! Original-context constructor and read-only standard-substruct methods.
use super::*;
use runmat_builtins::{
    BuiltinCatalogEntry, LSQLIN_WARM_SUBSASGN_CATALOG_ENTRY, LSQLIN_WARM_SUBSREF_CATALOG_ENTRY,
};
async fn start(
    entry: &BuiltinCatalogEntry,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<ScalarCooperation> {
    crate::output_count::validate_bounded_outputs_in(
        entry.descriptor,
        &OPTIMWARMSTART_ERROR_INVALID_INPUT,
        entry.identity.name,
        runtime,
    )?;
    Policy {
        identity: entry.identity,
        invalid: &OPTIMWARMSTART_ERROR_INVALID_INPUT,
    }
    .cooperation(runtime)
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optimwarmstart",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimwarmstart::bindings"
)]
async fn constructor(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let mut c = start(&OPTIMWARMSTART_CATALOG_ENTRY, runtime.as_ref()).await?;
    super::constructor::create(args.view(), &mut c).await
}
#[runmat_macros::runtime_builtin(
    name = "LsqlinWarmStart.subsref",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimwarmstart::bindings"
)]
async fn read(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    read_selected(
        &LSQLIN_WARM_SUBSREF_CATALOG_ENTRY,
        Kind::LeastSquares,
        runtime,
        args,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "LsqlinWarmStart.subsasgn",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimwarmstart::bindings"
)]
async fn write(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    write_selected(
        &LSQLIN_WARM_SUBSASGN_CATALOG_ENTRY,
        Kind::LeastSquares,
        runtime,
        args,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "QuadprogWarmStart.subsref",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimwarmstart::bindings"
)]
async fn read_quadratic(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    read_selected(
        &runmat_builtins::QUADPROG_WARM_SUBSREF_CATALOG_ENTRY,
        Kind::Quadratic,
        runtime,
        args,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "QuadprogWarmStart.subsasgn",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimwarmstart::bindings"
)]
async fn write_quadratic(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    write_selected(
        &runmat_builtins::QUADPROG_WARM_SUBSASGN_CATALOG_ENTRY,
        Kind::Quadratic,
        runtime,
        args,
    )
    .await
}
async fn read_selected(
    entry: &BuiltinCatalogEntry,
    kind: Kind,
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let mut c = start(entry, runtime.as_ref()).await?;
    if args.len() != 2 {
        return Err(policy().invalid("subsref expects a warm start and standard substruct"));
    }
    let _roots = policy().roots(args.iter(), c.runtime())?;
    let Value::Object(object) = &args[0] else {
        return Err(policy().invalid("expected a native warm start"));
    };
    let data = checked(object, kind, c.runtime())?;
    let path = crate::object::indexing::parse_standard_substruct(&args[1])?;
    super::properties::read(data, path, &mut c).await
}
async fn write_selected(
    entry: &BuiltinCatalogEntry,
    kind: Kind,
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let c = start(entry, runtime.as_ref()).await?;
    if args.len() != 3 {
        return Err(policy().invalid("subsasgn expects warm start, substruct and replacement"));
    }
    let Value::Object(object) = &args[0] else {
        return Err(policy().invalid("expected a native warm start"));
    };
    let _data = checked(object, kind, c.runtime())?;
    Err(crate::build_runtime_error(
        "warm-start properties are read-only; recreate with optimwarmstart",
    )
    .with_builtin(entry.identity.name)
    .with_identifier(
        runmat_builtins::OPTIMWARMSTART_ERROR_READ_ONLY
            .identifier
            .expect("declared read-only identifier"),
    )
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
    .build())
}
