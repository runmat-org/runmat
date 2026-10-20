use super::*;

pub(in crate::builtins::stats::summary) async fn fit_sample_distribution(
    data: Value,
    distname: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let admitted = admission::fit(&data, &distname, rest.view())?;
    ensure_fitdist_integer_extensions(&data, admitted.frequency)?;
    ensure_fitdist_resident_fallback_extension(&data, admitted.frequency)?;
    let sample_tensor = fitdist_value_to_tensor(data, "x").await?;
    let sample = parse_sample(sample_tensor, parse_fit_options(admitted.frequency).await?)?;
    let fit = fit_distribution(admitted.kind, &sample)?;
    Ok(Value::Object(distribution_object(&fit)?))
}
