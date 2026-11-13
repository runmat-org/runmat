use super::*;
use runmat_builtins::{
    NORMINV_INTEGER_MU_EXTENSION as INTEGER_MU_EXTENSION,
    NORMINV_INTEGER_SIGMA_EXTENSION as INTEGER_SIGMA_EXTENSION,
    NORMINV_INTEGER_X_EXTENSION as INTEGER_X_EXTENSION,
    NORMINV_LOGICAL_INPUT_EXTENSION as LOGICAL_INPUT_EXTENSION,
};

#[runtime_builtin(
    name = "norminv",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::distributions::norminv"
)]
pub(crate) async fn norminv_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::ensure_normal_extensions(
        "norminv",
        &value,
        rest.view(),
        [
            &INTEGER_X_EXTENSION,
            &INTEGER_MU_EXTENSION,
            &INTEGER_SIGMA_EXTENSION,
        ],
        &LOGICAL_INPUT_EXTENSION,
    )?;
    let args = super::normal_args("norminv", value, rest, DistributionKind::NormalInv).await?;
    let data = args
        .x
        .iter()
        .zip(args.mu.iter())
        .zip(args.sigma.iter())
        .map(|((p, mu), sigma)| super::norminv_scalar(*p, *mu, *sigma))
        .collect();
    args.output.finish("norminv", args.shape, data)
}
