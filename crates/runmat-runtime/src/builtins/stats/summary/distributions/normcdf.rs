use super::*;
use super::*;
normal_descriptor!("normcdf", NORMAL_CDF_SIGNATURES);
normal_integer_surface!(
    "normcdf",
    "normcdf-integer-x",
    "RunMat:compatibility:NormcdfIntegerXExtension",
    "normcdf-integer-mu",
    "RunMat:compatibility:NormcdfIntegerMuExtension",
    "normcdf-integer-sigma",
    "RunMat:compatibility:NormcdfIntegerSigmaExtension",
    "normcdf-logical-input",
    "RunMat:compatibility:NormcdfLogicalInputExtension"
);

#[runtime_builtin(
    name = "normcdf",
    category = "stats/summary",
    summary = "Evaluate the normal cumulative distribution function.",
    keywords = "normcdf,normal,gaussian,cdf,statistics",
    type_resolver(super::normal_type),
    descriptor(self::DESCRIPTOR),
    extensions(self::EXTENSIONS),
    integer_capabilities(self::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::summary::distributions::normcdf"
)]
pub(crate) async fn normcdf_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_extensions(&value, &rest)?;
    let args = super::normal_args("normcdf", value, rest, true).await?;
    let data = args
        .x
        .iter()
        .zip(args.mu.iter())
        .zip(args.sigma.iter())
        .map(|((x, mu), sigma)| {
            if args.upper {
                super::normcdf_upper_scalar(*x, *mu, *sigma)
            } else {
                super::normcdf_scalar(*x, *mu, *sigma)
            }
        })
        .collect();
    args.output.finish("normcdf", args.shape, data)
}
