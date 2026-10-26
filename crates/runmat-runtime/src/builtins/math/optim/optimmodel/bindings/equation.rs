//! Actual optimization problem constructor and native property protocol.
use super::*;
fn policy(entry: &runmat_builtins::BuiltinCatalogEntry) -> Policy {
    Policy {
        identity: entry.identity,
        invalid: &runmat_builtins::EQNPROBLEM_ERROR_INVALID_INPUT,
    }
}
fn validate(
    entry: &runmat_builtins::BuiltinCatalogEntry,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    crate::output_count::validate_bounded_outputs_in(
        entry.descriptor,
        &runmat_builtins::EQNPROBLEM_ERROR_INVALID_INPUT,
        entry.identity.name,
        runtime,
    )
}

#[runmat_macros::runtime_builtin(
    name = "eqnproblem",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::equation"
)]
async fn constructor(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::EQNPROBLEM_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    super::super::equation::create(args.view(), policy, &mut cooperation).await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.EquationProblem.subsref",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::equation"
)]
async fn read(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::EQNPROBLEM_SUBSREF_CATALOG_ENTRY;
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
    super::super::equation::read(&args[0], &path, policy, &mut cooperation).await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.EquationProblem.subsasgn",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::equation"
)]
async fn assign(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::EQNPROBLEM_SUBSASGN_CATALOG_ENTRY;
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
    super::super::equation::assign(&args[0], &path, &args[2], policy, &mut cooperation).await
}

#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.EquationProblem.solve",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::equation"
)]
async fn solve(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    let entry = &runmat_builtins::EQNPROBLEM_SOLVE_METHOD_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    crate::builtins::math::optim::common::native_class_registration::ensure(
        runmat_builtins::EQUATION_PROBLEM_CLASS,
        &[],
        policy,
        cooperation.runtime(),
    )?;
    super::super::equation::solve(args.view(), policy, &mut cooperation).await
}
