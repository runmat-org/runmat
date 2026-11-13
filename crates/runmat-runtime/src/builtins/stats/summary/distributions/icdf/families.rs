use super::*;

pub(super) fn parse_distribution_name(value: &Value) -> BuiltinResult<IcdfDistribution> {
    let Some(keyword) = crate::builtins::common::random_args::keyword_of(value)? else {
        return Err(super::normal_error(
            "icdf",
            "icdf: distribution name must be a string scalar",
        ));
    };
    IcdfDistribution::from_keyword(&keyword).ok_or_else(|| {
        super::normal_error(
            "icdf",
            format!("icdf: unsupported distribution '{keyword}'"),
        )
    })
}

pub(super) async fn normal_icdf(
    p: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let args = super::normal_args("icdf", p, rest, DistributionKind::GenericInv).await?;
    let data = args
        .x
        .iter()
        .zip(args.mu.iter())
        .zip(args.sigma.iter())
        .map(|((p, mu), sigma)| super::norminv_scalar(*p, *mu, *sigma))
        .collect();
    super::finish(args.shape, data)
}

pub(super) async fn student_t_icdf(
    p: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let args = super::t_args("icdf", p, rest).await?;
    let data = args
        .x
        .iter()
        .zip(args.nu.iter())
        .map(|(p, nu)| distribution_math::student_t_inv(*p, *nu))
        .collect();
    super::finish(args.shape, data)
}

pub(super) async fn weibull_icdf(
    p: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let args = super::three_args("icdf", p, rest, Some((1.0, 1.0))).await?;
    let data = args
        .x
        .iter()
        .zip(args.a.iter())
        .zip(args.b.iter())
        .map(|((p, a), b)| super::wblinv_scalar(*p, *a, *b))
        .collect();
    super::finish(args.shape, data)
}

pub(super) async fn chi_square_icdf(
    p: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let args = super::t_args("icdf", p, rest).await?;
    let data = args
        .x
        .iter()
        .zip(args.nu.iter())
        .map(|(p, nu)| super::chi2inv_scalar(*p, *nu))
        .collect();
    super::finish(args.shape, data)
}

pub(super) async fn binomial_icdf(
    p: Value,
    rest: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Value> {
    let args = super::three_args("icdf", p, rest, None).await?;
    let data = args
        .x
        .iter()
        .zip(args.a.iter())
        .zip(args.b.iter())
        .map(|((probability, n), success_probability)| {
            super::binoinv_scalar(*probability, *n, *success_probability, || {
                super::check_cancellation(runtime)
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    super::finish(args.shape, data)
}

pub(super) async fn gamma_icdf(
    p: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let args = super::three_args("icdf", p, rest, None).await?;
    let data = args
        .x
        .iter()
        .zip(args.a.iter())
        .zip(args.b.iter())
        .map(|((p, shape), scale)| super::gaminv_scalar(*p, *shape, *scale))
        .collect();
    super::finish(args.shape, data)
}

pub(super) async fn exponential_icdf(
    p: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let args = one_param_args(IcdfDistribution::Exponential, p, rest).await?;
    let data = args
        .x
        .iter()
        .zip(args.param.iter())
        .map(|(p, mu)| super::expinv_scalar(*p, *mu))
        .collect();
    super::finish(args.shape, data)
}

pub(super) async fn poisson_icdf(
    p: Value,
    rest: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Value> {
    let args = one_param_args(IcdfDistribution::Poisson, p, rest).await?;
    let data = args
        .x
        .iter()
        .zip(args.param.iter())
        .map(|(p, lambda)| {
            super::poissinv_scalar(*p, *lambda, || super::check_cancellation(runtime))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    super::finish(args.shape, data)
}

pub(super) async fn uniform_icdf(
    p: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let args = super::three_args("icdf", p, rest, Some((0.0, 1.0))).await?;
    let data = args
        .x
        .iter()
        .zip(args.a.iter())
        .zip(args.b.iter())
        .map(|((p, a), b)| super::unifinv_scalar(*p, *a, *b))
        .collect();
    super::finish(args.shape, data)
}

pub(super) async fn lognormal_icdf(
    p: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let args = super::normal_args("icdf", p, rest, DistributionKind::GenericInv).await?;
    let data = args
        .x
        .iter()
        .zip(args.mu.iter())
        .zip(args.sigma.iter())
        .map(|((p, mu), sigma)| super::logninv_scalar(*p, *mu, *sigma))
        .collect();
    super::finish(args.shape, data)
}

pub(super) async fn beta_icdf(
    p: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let args = super::three_args("icdf", p, rest, None).await?;
    let data = args
        .x
        .iter()
        .zip(args.a.iter())
        .zip(args.b.iter())
        .map(|((p, a), b)| super::betainv_scalar(*p, *a, *b))
        .collect();
    super::finish(args.shape, data)
}

pub(super) async fn f_icdf(p: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args = super::three_args("icdf", p, rest, None).await?;
    let data = args
        .x
        .iter()
        .zip(args.a.iter())
        .zip(args.b.iter())
        .map(|((p, v1), v2)| super::finv_scalar(*p, *v1, *v2))
        .collect();
    super::finish(args.shape, data)
}

struct OneParamArgs {
    x: Vec<f64>,
    param: Vec<f64>,
    shape: Vec<usize>,
}

async fn one_param_args(
    distribution: IcdfDistribution,
    p: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<OneParamArgs> {
    if rest.len() != 1 {
        return Err(super::normal_error(
            "icdf",
            format!("icdf: {distribution:?} distribution expects one parameter"),
        ));
    }
    let x = super::value_to_tensor("icdf", p).await?;
    let param = super::value_to_tensor("icdf", rest[0].clone()).await?;
    let (x, param, shape) = super::broadcast_pair("icdf", &x, &param)?;
    Ok(OneParamArgs { x, param, shape })
}
