//! Actual optimization problem constructor and native property protocol.
use super::*;
fn policy(entry: &runmat_builtins::BuiltinCatalogEntry) -> Policy {
    Policy {
        identity: entry.identity,
        invalid: &runmat_builtins::OPTIMPROBLEM_ERROR_INVALID_INPUT,
    }
}
fn validate(
    entry: &runmat_builtins::BuiltinCatalogEntry,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    crate::output_count::validate_bounded_outputs_in(
        entry.descriptor,
        &runmat_builtins::OPTIMPROBLEM_ERROR_INVALID_INPUT,
        entry.identity.name,
        runtime,
    )
}

#[runmat_macros::runtime_builtin(
    name = "optimproblem",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::problem"
)]
async fn constructor(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMPROBLEM_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::problem::create(args.view(), policy, &mut cooperation).await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationProblem.subsref",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::problem"
)]
async fn read(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMPROBLEM_SUBSREF_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    if args.len() != 2 {
        return Err(
            policy.invalid("problem subsref expects receiver and standard subscript descriptor")
        );
    }
    let _inputs = policy.roots(args.iter(), cooperation.runtime())?;
    let path = crate::object::indexing::parse_standard_substruct(&args[1])?;
    super::super::problem::read(&args[0], &path, policy, &mut cooperation).await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationProblem.subsasgn",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::problem"
)]
async fn assign(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMPROBLEM_SUBSASGN_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    if args.len() != 3 {
        return Err(policy.invalid(
            "problem subsasgn expects receiver, standard subscript descriptor and value",
        ));
    }
    let _inputs = policy.roots(args.iter(), cooperation.runtime())?;
    let path = crate::object::indexing::parse_standard_substruct(&args[1])?;
    super::super::problem::assign(&args[0], &path, &args[2], policy, &mut cooperation).await
}

async fn solve_entry(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
    entry: &runmat_builtins::BuiltinCatalogEntry,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    if matches!(args.get(0), Some(Value::Object(object)) if object.class_name().is(runmat_builtins::EQUATION_PROBLEM_CLASS))
    {
        crate::builtins::math::optim::common::native_class_registration::ensure(
            runmat_builtins::EQUATION_PROBLEM_CLASS,
            &[],
            policy,
            cooperation.runtime(),
        )?;
        return super::super::equation::solve(args.view(), policy, &mut cooperation).await;
    }
    super::super::registration::ensure(
        standard::OPTIMIZATION_PROBLEM,
        policy,
        cooperation.runtime(),
    )?;
    super::super::problem::solve(args.view(), policy, &mut cooperation).await
}
#[runmat_macros::runtime_builtin(
    name = "solve",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::problem"
)]
async fn solve(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    solve_entry(
        runtime,
        args,
        &runmat_builtins::OPTIMPROBLEM_SOLVE_CATALOG_ENTRY,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationProblem.solve",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::problem"
)]
async fn solve_method(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    solve_entry(
        runtime,
        args,
        &runmat_builtins::OPTIMPROBLEM_SOLVE_METHOD_CATALOG_ENTRY,
    )
    .await
}
