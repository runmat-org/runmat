//! Minimax public callable, sharing the constrained epigraph owner.
use super::common::multiobjective::Kind;
use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
#[runtime_builtin(
    name = "fminimax",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::fminimax"
)]
async fn fminimax(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    super::fmincon::multiobjective::execute(
        runtime.as_ref(),
        args,
        super::fmincon::multiobjective::Definition {
            kind: Kind::Minimax,
            descriptor: &runmat_builtins::FMINIMAX_DESCRIPTOR,
            error: &runmat_builtins::FMINIMAX_ERROR_INVALID_ARGUMENT,
            numeric: &runmat_builtins::FMINIMAX_NUMERIC_EXTENSION,
            single: &runmat_builtins::FMINIMAX_SINGLE_EXTENSION,
            resident: &runmat_builtins::FMINIMAX_RESIDENT_EXTENSION,
            integer_option: &runmat_builtins::FMINIMAX_INTEGER_OPTION_EXTENSION,
            sparse: &runmat_builtins::FMINIMAX_SPARSE_EXTENSION,
        },
    )
    .await
}
