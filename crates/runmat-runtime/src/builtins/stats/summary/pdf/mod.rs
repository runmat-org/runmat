use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "pdf",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::pdf"
)]
pub(crate) async fn pdf_builtin(
    distribution: Value,
    x: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::fitted_distribution::evaluate_pdf_entry(distribution, x, rest).await
}
