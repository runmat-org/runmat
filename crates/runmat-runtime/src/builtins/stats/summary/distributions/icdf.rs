use super::*;
use runmat_builtins::{
    InverseDistributionFamily as IcdfDistribution,
    ICDF_INTEGER_INPUT_EXTENSION as INTEGER_INPUT_EXTENSION,
    ICDF_LOGICAL_INPUT_EXTENSION as LOGICAL_INPUT_EXTENSION,
};

#[runtime_builtin(
    name = "icdf",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::summary::distributions::icdf"
)]
pub(crate) async fn icdf_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    name: Value,
    p: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    icdf_in_runtime(runtime.as_ref(), name, p, rest).await
}

#[cfg(test)]
pub(crate) async fn icdf_builtin(name: Value, p: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest =
        crate::execution::RootedValues::new(rest).map_err(|error| error.into_runtime_error())?;
    icdf_in_runtime(None, name, p, rest.into()).await
}

pub(super) fn check_cancellation(
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    if let Some(runtime) = runtime.filter(|runtime| runtime.is_cancelled()) {
        return Err(build_runtime_error("icdf: execution was cancelled")
            .with_identifier(format!("{}:Cancelled", runtime.error_namespace()))
            .with_builtin("icdf")
            .build());
    }
    Ok(())
}

async fn icdf_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    name: Value,
    p: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    check_cancellation(runtime)?;
    if matches!(name, Value::Object(_)) {
        if !rest.is_empty() {
            return Err(super::normal_error(
                "icdf",
                "icdf: fitted distribution object form accepts exactly two inputs",
            ));
        }
        let fitted = crate::builtins::stats::summary::fitted_distribution::admit_icdf_probability_distribution(&name)?;
        super::numeric_inputs("icdf", &[&p], super::NumericShapePolicy::Broadcast)?;
        let admission = IcdfAdmission::new(&p, rest.view())?;
        let p = gather_input(p, runtime).await?;
        ensure_exact_integer_boundary(&p, "p")?;
        let output =
            crate::builtins::stats::summary::fitted_distribution::icdf_probability_distribution(
                fitted, p, runtime,
            )
            .await?;
        check_cancellation(runtime)?;
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
    let admission = IcdfAdmission::new(&p, rest.view())?;
    let p = gather_input(p, runtime).await?;
    let gathered_rest = rest
        .try_map_preserving_roots_async(rest.len(), runtime, |index| {
            gather_input(rest[index].clone(), runtime)
        })
        .await?;
    drop(rest);
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
        IcdfDistribution::Binomial => binomial_icdf(p, gathered_rest, runtime).await,
        IcdfDistribution::Gamma => gamma_icdf(p, gathered_rest).await,
        IcdfDistribution::Exponential => exponential_icdf(p, gathered_rest).await,
        IcdfDistribution::Poisson => poisson_icdf(p, gathered_rest, runtime).await,
        IcdfDistribution::Uniform => uniform_icdf(p, gathered_rest).await,
        IcdfDistribution::Lognormal => lognormal_icdf(p, gathered_rest).await,
        IcdfDistribution::Beta => beta_icdf(p, gathered_rest).await,
        IcdfDistribution::F => f_icdf(p, gathered_rest).await,
    }?;
    check_cancellation(runtime)?;
    admission.finish(output)
}

mod admission;
mod families;

use admission::{ensure_exact_integer_boundary, gather_input, IcdfAdmission};
use families::*;
