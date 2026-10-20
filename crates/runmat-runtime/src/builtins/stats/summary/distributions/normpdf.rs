use super::*;
use runmat_builtins::{
    NORMPDF_INTEGER_MU_EXTENSION as INTEGER_MU_EXTENSION,
    NORMPDF_INTEGER_SIGMA_EXTENSION as INTEGER_SIGMA_EXTENSION,
    NORMPDF_INTEGER_X_EXTENSION as INTEGER_X_EXTENSION,
    NORMPDF_LOGICAL_INPUT_EXTENSION as LOGICAL_INPUT_EXTENSION,
};

#[runtime_builtin(
    name = "normpdf",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::distributions::normpdf"
)]
pub(crate) async fn normpdf_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::ensure_normal_extensions(
        "normpdf",
        &value,
        rest.view(),
        [
            &INTEGER_X_EXTENSION,
            &INTEGER_MU_EXTENSION,
            &INTEGER_SIGMA_EXTENSION,
        ],
        &LOGICAL_INPUT_EXTENSION,
    )?;
    let args = super::normal_args("normpdf", value, rest, DistributionKind::NormalPdf).await?;
    let data = args
        .x
        .iter()
        .zip(args.mu.iter())
        .zip(args.sigma.iter())
        .map(|((x, mu), sigma)| super::normpdf_scalar(*x, *mu, *sigma))
        .collect();
    args.output.finish("normpdf", args.shape, data)
}
