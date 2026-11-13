//! Actual enum method entries, retaining original class/context/cancellation.
use super::*;
#[derive(Clone, Copy)]
enum Operation {
    Number,
    Text,
    Equal,
    NotEqual,
}
async fn execute(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
    entry: &runmat_builtins::BuiltinCatalogEntry,
    operation: Operation,
) -> BuiltinResult<RootedValue> {
    crate::output_count::validate_bounded_outputs_in(
        entry.descriptor,
        &runmat_builtins::OPTIMPROBLEM_ERROR_INVALID_INPUT,
        entry.identity.name,
        runtime.as_ref(),
    )?;
    let policy = Policy {
        identity: entry.identity,
        invalid: &runmat_builtins::OPTIMPROBLEM_ERROR_INVALID_INPUT,
    };
    let cooperation = policy.cooperation(runtime.as_ref()).await?;
    let count = if matches!(operation, Operation::Equal | Operation::NotEqual) {
        2
    } else {
        1
    };
    if args.len() != count {
        return Err(policy.invalid("invalid termination method arity"));
    }
    let _roots = policy.roots(args.iter(), cooperation.runtime())?;
    let first = super::super::exitflag::read(&args[0], policy, cooperation.runtime())?;
    let value = match operation {
        Operation::Number => Value::Num(f64::from(first.numeric())),
        Operation::Text => Value::string(crate::call::argument_text::copy_str(
            first.name(),
            cooperation.runtime(),
        )?),
        Operation::Equal | Operation::NotEqual => {
            let second = super::super::exitflag::read(&args[1], policy, cooperation.runtime())?;
            Value::Bool(if matches!(operation, Operation::Equal) {
                first == second
            } else {
                first != second
            })
        }
    };
    RootedValue::in_optional_context(value, cooperation.runtime())
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExitflag.double",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::exitflag"
)]
async fn number(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    execute(
        runtime,
        args,
        &runmat_builtins::OPTIMEXITFLAG_DOUBLE_CATALOG_ENTRY,
        Operation::Number,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExitflag.string",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::exitflag"
)]
async fn text(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    execute(
        runtime,
        args,
        &runmat_builtins::OPTIMEXITFLAG_STRING_CATALOG_ENTRY,
        Operation::Text,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExitflag.eq",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::exitflag"
)]
async fn equal(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    execute(
        runtime,
        args,
        &runmat_builtins::OPTIMEXITFLAG_EQ_CATALOG_ENTRY,
        Operation::Equal,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExitflag.ne",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::exitflag"
)]
async fn not_equal(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    execute(
        runtime,
        args,
        &runmat_builtins::OPTIMEXITFLAG_NE_CATALOG_ENTRY,
        Operation::NotEqual,
    )
    .await
}
