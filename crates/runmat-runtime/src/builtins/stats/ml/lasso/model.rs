use super::*;

pub(super) fn lasso_compute(
    x: Value,
    y: Value,
    mut options: LassoOptions,
) -> BuiltinResult<FitResult> {
    let x = value_to_real_tensor("X", x)?;
    let y_values = value_to_real_vector("y", y)?;
    let (rows, cols) = crate::builtins::common::linalg::matrix_dimensions_for("X", x.shape())
        .map_err(|detail| invalid_argument(format!("lasso: {detail}")))?;
    if rows == 0 || cols == 0 {
        return Err(invalid_argument(
            "lasso: X must be a nonempty numeric matrix",
        ));
    }
    if y_values.len() != rows {
        return Err(invalid_argument(format!(
            "lasso: y length {} must match the number of rows in X ({rows})",
            y_values.len()
        )));
    }
    let x_values = tensor::tensor_values_f64_cow(&x);
    if y_values.iter().any(|value| !value.is_finite())
        || x_values.iter().any(|value| !value.is_finite())
    {
        return Err(invalid_argument(
            "lasso: X and y must contain finite real values",
        ));
    }
    if options
        .predictor_names
        .as_ref()
        .is_some_and(|names| names.len() != cols)
    {
        return Err(invalid_argument(
            "lasso: PredictorNames must contain one name per predictor",
        ));
    }
    if options
        .weights
        .as_ref()
        .is_some_and(|weights| weights.len() != rows)
    {
        return Err(invalid_argument(
            "lasso: Weights vector length must match the number of rows in X",
        ));
    }
    if !options.intercept {
        options.standardize = false;
    }

    let prepared = prepare_data(&x, &y_values, &options)?;
    let lambdas = match &options.lambda {
        Some(lambda) => sorted_unique_lambdas(lambda)?,
        None => default_lambda_sequence(&prepared, &options)?,
    };
    let fit_path = fit_path(&prepared, &lambdas, &options)?;
    let fit_path = apply_df_limit(fit_path, options.df_max)?;
    let cv = match options.cv {
        CvSpec::Resubstitution => None,
        CvSpec::KFold(k) => Some(cross_validate(
            &x,
            &y_values,
            &fit_path.lambdas,
            &options,
            k,
        )?),
    };
    let b = coefficients_value(&fit_path)?;
    let fit_info = fit_info_value(&fit_path, &options, cv)?;
    Ok(FitResult { b, fit_info })
}
