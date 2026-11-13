//! Semi-infinite constrained optimization through the typed dynamic-mesh owner.
pub use runmat_builtins::FSEMINF_DESCRIPTOR;
use runmat_macros::runtime_builtin;
#[runtime_builtin(
    name = "fseminf",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::fseminf"
)]
async fn registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValueSequence> {
    super::fmincon::semi_infinite::execute(runtime.as_ref(), args).await
}
