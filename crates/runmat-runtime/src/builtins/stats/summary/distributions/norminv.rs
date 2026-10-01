use super::*;
use super::*;
normal_descriptor!("norminv", NORMAL_INV_SIGNATURES);
normal_integer_surface!(
    "norminv",
    "norminv-integer-p",
    "RunMat:compatibility:NorminvIntegerPExtension",
    "norminv-integer-mu",
    "RunMat:compatibility:NorminvIntegerMuExtension",
    "norminv-integer-sigma",
    "RunMat:compatibility:NorminvIntegerSigmaExtension",
    "norminv-logical-input",
    "RunMat:compatibility:NorminvLogicalInputExtension"
);

#[runtime_builtin(
    name = "norminv",
    category = "stats/summary",
    summary = "Evaluate the inverse normal cumulative distribution function.",
    keywords = "norminv,normal,gaussian,inverse,cdf,statistics",
    type_resolver(super::normal_type),
    descriptor(self::DESCRIPTOR),
    extensions(self::EXTENSIONS),
    integer_capabilities(self::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::summary::distributions::norminv"
)]
pub(crate) async fn norminv_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_extensions(&value, &rest)?;
    let args = super::normal_args("norminv", value, rest, false).await?;
    let data = args
        .x
        .iter()
        .zip(args.mu.iter())
        .zip(args.sigma.iter())
        .map(|((p, mu), sigma)| super::norminv_scalar(*p, *mu, *sigma))
        .collect();
    args.output.finish("norminv", args.shape, data)
}
