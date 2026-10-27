//! Goal-attainment public callable, sharing the constrained epigraph owner.
use super::common::multiobjective::Kind;
use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
#[runtime_builtin(
    name = "fgoalattain",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::fgoalattain"
)]
async fn fgoalattain(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    super::fmincon::multiobjective::execute(
        runtime.as_ref(),
        args,
        super::fmincon::multiobjective::Definition {
            kind: Kind::GoalAttain,
            descriptor: &runmat_builtins::FGOALATTAIN_DESCRIPTOR,
            error: &runmat_builtins::FGOALATTAIN_ERROR_INVALID_ARGUMENT,
            numeric: &runmat_builtins::FGOALATTAIN_NUMERIC_EXTENSION,
            single: &runmat_builtins::FGOALATTAIN_SINGLE_EXTENSION,
            resident: &runmat_builtins::FGOALATTAIN_RESIDENT_EXTENSION,
            integer_option: &runmat_builtins::FGOALATTAIN_INTEGER_OPTION_EXTENSION,
            sparse: &runmat_builtins::FGOALATTAIN_SPARSE_EXTENSION,
        },
    )
    .await
}
