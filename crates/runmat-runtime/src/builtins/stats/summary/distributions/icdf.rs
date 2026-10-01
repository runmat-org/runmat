use super::*;
use runmat_builtins::{
    InverseDistributionFamily as IcdfDistribution,
    ICDF_INTEGER_INPUT_EXTENSION as INTEGER_INPUT_EXTENSION,
    ICDF_LOGICAL_INPUT_EXTENSION as LOGICAL_INPUT_EXTENSION,
};

#[runtime_builtin(
    name = "icdf",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::distributions::icdf"
)]
pub(crate) async fn icdf_builtin(name: Value, p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if matches!(name, Value::Object(_)) {
        if !rest.is_empty() {
            return Err(super::normal_error(
                "icdf",
                "icdf: fitted distribution object form accepts exactly two inputs",
            ));
        }
        crate::builtins::stats::summary::fitdist::validate_icdf_probability_distribution(&name)?;
        super::numeric_inputs("icdf", &[&p], super::NumericShapePolicy::Broadcast)?;
        let admission = IcdfAdmission::new(&p, &[])?;
        let p = gather_input(p).await?;
        ensure_exact_integer_boundary(&p, "p")?;
        let output =
            crate::builtins::stats::summary::fitdist::icdf_probability_distribution(name, p)
                .await?;
        return admission.finish(output);
    }

    let distribution = parse_distribution_name(&name)?;
    if !distribution.accepts_parameter_count(rest.len()) {
        return Err(super::normal_error(
            "icdf",
            "icdf: incorrect parameter count for this distribution",
        ));
    }
    let inputs = std::iter::once(&p).chain(rest.iter()).collect::<Vec<_>>();
    super::numeric_inputs("icdf", &inputs, super::NumericShapePolicy::Broadcast)?;
    let admission = IcdfAdmission::new(&p, &rest)?;
    let p = gather_input(p).await?;
    let mut gathered_rest = Vec::with_capacity(rest.len());
    for value in rest {
        gathered_rest.push(gather_input(value).await?);
    }
    for (label, value) in
        std::iter::once(("p", &p)).chain(gathered_rest.iter().map(|value| ("parameter", value)))
    {
        ensure_exact_integer_boundary(value, label)?;
    }
    let output = match distribution {
        IcdfDistribution::Normal => normal_icdf(p, gathered_rest).await,
        IcdfDistribution::StudentT => student_t_icdf(p, gathered_rest).await,
        IcdfDistribution::Weibull => weibull_icdf(p, gathered_rest).await,
        IcdfDistribution::ChiSquare => chi_square_icdf(p, gathered_rest).await,
        IcdfDistribution::Binomial => binomial_icdf(p, gathered_rest).await,
        IcdfDistribution::Gamma => gamma_icdf(p, gathered_rest).await,
        IcdfDistribution::Exponential => exponential_icdf(p, gathered_rest).await,
        IcdfDistribution::Poisson => poisson_icdf(p, gathered_rest).await,
        IcdfDistribution::Uniform => uniform_icdf(p, gathered_rest).await,
        IcdfDistribution::Lognormal => lognormal_icdf(p, gathered_rest).await,
        IcdfDistribution::Beta => beta_icdf(p, gathered_rest).await,
        IcdfDistribution::F => f_icdf(p, gathered_rest).await,
    }?;
    admission.finish(output)
}

mod admission;
mod families;

use admission::{ensure_exact_integer_boundary, gather_input, IcdfAdmission};
use families::*;
