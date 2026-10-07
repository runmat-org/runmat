use super::*;

pub(super) fn lassoglm_compute(
    x: Value,
    y: Value,
    distribution: Distribution,
    options: Options,
) -> BuiltinResult<FitResult> {
    let x = value_to_real_tensor("X", x)?;
    let (rows, cols) = crate::builtins::common::linalg::matrix_dimensions_for("X", x.shape())
        .map_err(|detail| invalid(format!("lassoglm: {detail}")))?;
    if rows == 0 || cols == 0 {
        return Err(invalid("lassoglm: X must be a nonempty numeric matrix"));
    }
    validate_distribution_options(distribution, &options)?;
    let response = response_values(y, distribution, rows, &options)?;
    if response.0.len() != rows {
        return Err(invalid(format!(
            "lassoglm: Y length {} must match the number of rows in X ({rows})",
            response.0.len()
        )));
    }
    if tensor::tensor_values_f64_cow(&x)
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(invalid("lassoglm: X must contain finite real values"));
    }
    if options
        .predictor_names
        .as_ref()
        .is_some_and(|names| names.len() != cols)
    {
        return Err(invalid(
            "lassoglm: PredictorNames must contain one name per predictor",
        ));
    }
    if options
        .weights
        .as_ref()
        .is_some_and(|weights| weights.len() != rows)
    {
        return Err(invalid(
            "lassoglm: Weights vector length must match the number of rows in X",
        ));
    }
    if options
        .offset
        .as_ref()
        .is_some_and(|offset| offset.len() != rows)
    {
        return Err(invalid(
            "lassoglm: Offset vector length must match the number of rows in X",
        ));
    }
    let prepared = prepare_data(&x, response, &options)?;
    let lambdas = match &options.lambda {
        Some(values) => sorted_unique_lambdas(values)?,
        None => default_lambda_sequence(&prepared, distribution, &options)?,
    };
    let fit_path = fit_path(&prepared, distribution, &lambdas, &options)?;
    let cv = match options.cv {
        CvSpec::Resubstitution => None,
        CvSpec::KFold(k) => Some(cross_validate(
            &x,
            &prepared,
            distribution,
            &fit_path.lambdas,
            &options,
            k,
        )?),
    };
    let b = coefficients_value(&fit_path)?;
    let fit_info = fit_info_value(&fit_path, distribution, &options, cv)?;
    Ok(FitResult { b, fit_info })
}
