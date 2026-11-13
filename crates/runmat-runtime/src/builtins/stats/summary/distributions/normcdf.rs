use super::*;
use runmat_builtins::{
    NORMCDF_INTEGER_MU_EXTENSION as INTEGER_MU_EXTENSION,
    NORMCDF_INTEGER_SIGMA_EXTENSION as INTEGER_SIGMA_EXTENSION,
    NORMCDF_INTEGER_X_EXTENSION as INTEGER_X_EXTENSION,
    NORMCDF_LOGICAL_INPUT_EXTENSION as LOGICAL_INPUT_EXTENSION,
};

#[runtime_builtin(
    name = "normcdf",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::distributions::normcdf"
)]
pub(crate) async fn normcdf_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::ensure_normal_extensions(
        "normcdf",
        &value,
        rest.view(),
        [
            &INTEGER_X_EXTENSION,
            &INTEGER_MU_EXTENSION,
            &INTEGER_SIGMA_EXTENSION,
        ],
        &LOGICAL_INPUT_EXTENSION,
    )?;
    let args = super::normal_args("normcdf", value, rest, DistributionKind::NormalCdf).await?;
    let data = args
        .x
        .iter()
        .zip(args.mu.iter())
        .zip(args.sigma.iter())
        .map(|((x, mu), sigma)| {
            if args.tail == DistributionTail::Upper {
                super::normcdf_upper_scalar(*x, *mu, *sigma)
            } else {
                super::normcdf_scalar(*x, *mu, *sigma)
            }
        })
        .collect();
    args.output.finish("normcdf", args.shape, data)
}
