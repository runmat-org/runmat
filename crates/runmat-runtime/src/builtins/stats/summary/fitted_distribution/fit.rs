use super::*;

pub(super) async fn fit_sample_distribution(
    data: Value,
    distname: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    admission::fit(&data, &distname, &rest)?;
    ensure_fitdist_integer_extensions(&data, &rest)?;
    ensure_fitdist_resident_fallback_extension(&data, &rest)?;
    let kind = parse_distribution_name(&distname)?;
    let sample_tensor = fitdist_value_to_tensor(data, "x").await?;
    let sample = parse_sample(sample_tensor, parse_fit_options(rest).await?)?;
    let fit = fit_distribution(kind, &sample)?;
    Ok(Value::Object(distribution_object(&fit)?))
}
