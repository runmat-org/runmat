use super::*;
use super::*;
normal_descriptor!("normpdf", NORMAL_SIGNATURES_X);
normal_integer_surface!(
    "normpdf",
    "normpdf-integer-x",
    "RunMat:compatibility:NormpdfIntegerXExtension",
    "normpdf-integer-mu",
    "RunMat:compatibility:NormpdfIntegerMuExtension",
    "normpdf-integer-sigma",
    "RunMat:compatibility:NormpdfIntegerSigmaExtension",
    "normpdf-logical-input",
    "RunMat:compatibility:NormpdfLogicalInputExtension"
);

#[runtime_builtin(
    name = "normpdf",
    category = "stats/summary",
    summary = "Evaluate the normal probability density function.",
    keywords = "normpdf,normal,gaussian,pdf,statistics",
    type_resolver(super::normal_type),
    descriptor(self::DESCRIPTOR),
    extensions(self::EXTENSIONS),
    integer_capabilities(self::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::summary::distributions::normpdf"
)]
pub(crate) async fn normpdf_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_extensions(&value, &rest)?;
    let args = super::normal_args("normpdf", value, rest, false).await?;
    let data = args
        .x
        .iter()
        .zip(args.mu.iter())
        .zip(args.sigma.iter())
        .map(|((x, mu), sigma)| super::normpdf_scalar(*x, *mu, *sigma))
        .collect();
    args.output.finish("normpdf", args.shape, data)
}
