use super::*;

/// Projects a dense predictor value into fitclinear's observation matrix.
///
/// Rank-one predictors are one observation with N predictors, while rank-zero
/// predictors have no matrix extent. This preserves the classification domain
/// contract rather than adopting Tensor's general matrix projection.
pub(super) fn classification_matrix_dimensions(tensor: &Tensor) -> (usize, usize) {
    match tensor.shape() {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}

pub(super) fn build_fit_spec(
    first: Value,
    rest: crate::call::FunctionArgumentView<'_>,
    arguments: FitclinearArguments,
) -> BuiltinResult<FitSpec> {
    if is_table_value(&first) {
        build_table_fit_spec(first, rest, arguments)
    } else {
        if rest.is_empty() {
            return Err(fitclinear_invalid(
                "fitclinear: matrix input requires response vector Y",
            ));
        }
        let y = rest[0].clone();
        let rest = rest.range(1..)?;
        build_matrix_fit_spec(first, y, rest, arguments)
    }
}

pub(super) fn is_table_value(value: &Value) -> bool {
    matches!(value, Value::Object(object) if is_tabular_object(object))
}

fn build_matrix_fit_spec(
    x_value: Value,
    y_value: Value,
    rest: crate::call::FunctionArgumentView<'_>,
    arguments: FitclinearArguments,
) -> BuiltinResult<FitSpec> {
    let mut options = FitOptions::default();
    parse_fit_options(rest, arguments.options, 1, &mut options)?;
    if let Some(WeightSpec::Variable(_)) = options.weights {
        return Err(fitclinear_invalid(
            "fitclinear: weights variable names require table input",
        ));
    }
    let x = normalize_observation_matrix(numeric_matrix_for_fit(x_value)?, &options)?;
    let (x_rows, x_cols) = classification_matrix_dimensions(&x);
    let labels = label_vector(y_value, "Y")?;
    if labels.labels.len() != x_rows {
        return Err(fitclinear_invalid(
            "fitclinear: length(Y) must match the number of observations in X",
        ));
    }
    let predictor_names = if let Some(names) = options.predictor_names.clone() {
        if names.len() != x_cols {
            return Err(fitclinear_invalid(format!(
                "fitclinear: PredictorNames must contain {} names",
                x_cols
            )));
        }
        names
    } else {
        (1..=x_cols).map(|idx| format!("x{idx}")).collect()
    };
    let response_name = options
        .response_name
        .clone()
        .unwrap_or_else(|| "Y".to_string());
    let weights = match &options.weights {
        Some(WeightSpec::Values(values)) => values.clone(),
        Some(WeightSpec::Variable(_)) => unreachable!("checked above"),
        None => vec![1.0; x_rows],
    };
    finalize_fit_spec(x, labels, predictor_names, response_name, weights, options)
}

fn build_table_fit_spec(
    first: Value,
    rest: crate::call::FunctionArgumentView<'_>,
    arguments: FitclinearArguments,
) -> BuiltinResult<FitSpec> {
    let object = match first {
        Value::Object(object) => object,
        _ => unreachable!("table checked"),
    };
    let variable_names = table_variable_names_from_object(&object)
        .map_err(|err| fitclinear_invalid(format!("fitclinear: {err}")))?;
    if variable_names.len() < 2 {
        return Err(fitclinear_invalid(
            "fitclinear: table input must contain at least one predictor and one response",
        ));
    }
    let variables =
        table_variables(&object).map_err(|err| fitclinear_invalid(format!("fitclinear: {err}")))?;
    let mut options = FitOptions::default();
    let mut response_name = variable_names
        .last()
        .cloned()
        .ok_or_else(|| fitclinear_invalid("fitclinear: response variable missing"))?;
    let mut external_y = None;
    let mut formula_predictors = None;
    if let Some(first_arg) = rest.first() {
        if arguments.option_start == 1 {
            if let Some(text) = scalar_text(first_arg)? {
                if text.contains('~') {
                    let parsed = parse_formula(&text, &variable_names)?;
                    response_name = parsed.0;
                    formula_predictors = Some(parsed.1);
                } else {
                    response_name = text;
                }
            } else {
                external_y = Some(label_vector(first_arg.clone(), "Y")?);
                response_name = options
                    .response_name
                    .clone()
                    .unwrap_or_else(|| "Y".to_string());
            }
        }
    }
    parse_fit_options(rest, arguments.options, 0, &mut options)?;
    if options.observations_in == ObservationsIn::Columns {
        return Err(fitclinear_invalid(
            "fitclinear: ObservationsIn='columns' is not supported for table input",
        ));
    }
    response_name = options.response_name.clone().unwrap_or(response_name);
    let has_external_y = external_y.is_some();
    let labels = if let Some(labels) = external_y {
        labels
    } else {
        let value = variables.fields.get(&response_name).ok_or_else(|| {
            fitclinear_invalid(format!(
                "fitclinear: response variable '{response_name}' is not in the table"
            ))
        })?;
        label_vector(value.clone(), &response_name)?
    };
    let predictor_names = if let Some(names) = options.predictor_names.clone() {
        names
    } else if let Some(names) = formula_predictors {
        names
    } else {
        let weight_variable = match &options.weights {
            Some(WeightSpec::Variable(name)) => Some(name.as_str()),
            _ => None,
        };
        variable_names
            .iter()
            .filter(|name| {
                (has_external_y || **name != response_name)
                    && weight_variable != Some(name.as_str())
            })
            .cloned()
            .collect()
    };
    if predictor_names.is_empty() {
        return Err(fitclinear_invalid(
            "fitclinear: at least one predictor variable is required",
        ));
    }
    let mut columns = Vec::with_capacity(predictor_names.len());
    for name in &predictor_names {
        let value = variables.fields.get(name).ok_or_else(|| {
            fitclinear_invalid(format!(
                "fitclinear: predictor variable '{name}' is not in the table"
            ))
        })?;
        let tensor = numeric_matrix_for_fit(value.clone()).map_err(|_| {
            fitclinear_invalid(format!(
                "fitclinear: predictor variable '{name}' must be numeric"
            ))
        })?;
        columns.push(vector_values(&tensor, name)?);
    }
    let x = columns_to_tensor(columns)?;
    let x_rows = classification_matrix_dimensions(&x).0;
    let weights = match &options.weights {
        Some(WeightSpec::Values(values)) => values.clone(),
        Some(WeightSpec::Variable(name)) => {
            let value = variables.fields.get(name).ok_or_else(|| {
                fitclinear_invalid(format!(
                    "fitclinear: weights variable '{name}' is not in the table"
                ))
            })?;
            numeric_vector(value, "Weights")?
        }
        None => vec![1.0; x_rows],
    };
    finalize_fit_spec(x, labels, predictor_names, response_name, weights, options)
}

fn finalize_fit_spec(
    x: Tensor,
    labels: LabelVector,
    predictor_names: Vec<String>,
    response_name: String,
    weights: Vec<f64>,
    mut options: FitOptions,
) -> BuiltinResult<FitSpec> {
    let (x_rows, x_cols) = classification_matrix_dimensions(&x);
    if labels.labels.len() != x_rows {
        return Err(fitclinear_invalid(
            "fitclinear: response length must match the number of observations",
        ));
    }
    if weights.len() != x_rows {
        return Err(fitclinear_invalid(
            "fitclinear: Weights length must match the number of observations",
        ));
    }
    if predictor_names.len() != x_cols {
        return Err(fitclinear_invalid(format!(
            "fitclinear: PredictorNames must contain {} names",
            x_cols
        )));
    }
    for weight in &weights {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(fitclinear_invalid(
                "fitclinear: weights must be nonnegative finite values",
            ));
        }
    }
    let class_names = if let Some(names) = options.class_names.clone() {
        validate_class_names(names, labels.kind)?
    } else {
        unique_labels(&labels.labels, labels.kind)
    };
    if class_names.len() != 2 {
        return Err(fitclinear_invalid(
            "fitclinear: exactly two classes are required",
        ));
    }
    let mut clean_rows = Vec::new();
    let mut clean_classes = Vec::new();
    let mut clean_weights = Vec::new();
    for row in 0..x_rows {
        let weight = weights[row];
        if weight == 0.0 || is_missing_label(&labels.labels[row]) {
            continue;
        }
        let class_index = class_names
            .iter()
            .position(|label| same_label(label, &labels.labels[row]))
            .ok_or_else(|| {
                fitclinear_invalid("fitclinear: response contains a class outside ClassNames")
            })?;
        let mut row_values = Vec::with_capacity(x_cols);
        let mut complete = true;
        for col in 0..x_cols {
            let value = x_value(&x, row, col);
            if value.is_infinite() {
                return Err(fitclinear_invalid(
                    "fitclinear: predictor values must not contain Inf",
                ));
            }
            if value.is_nan() {
                complete = false;
                break;
            }
            row_values.push(value);
        }
        if complete {
            clean_rows.push(row_values);
            clean_classes.push(class_index);
            clean_weights.push(weight);
        }
    }
    if clean_classes.is_empty() {
        return Err(fitclinear_invalid(
            "fitclinear: at least one complete weighted observation is required",
        ));
    }
    if !(clean_classes.contains(&0) && clean_classes.contains(&1)) {
        return Err(fitclinear_invalid(
            "fitclinear: both classes must be observed after removing missing rows",
        ));
    }
    let clean_x = rows_to_tensor(clean_rows)?;
    let (clean_rows, clean_cols) = classification_matrix_dimensions(&clean_x);
    let lambdas = options
        .lambdas
        .clone()
        .unwrap_or_else(|| vec![1.0 / clean_rows.max(1) as f64]);
    if lambdas.is_empty() {
        return Err(fitclinear_invalid(
            "fitclinear: Lambda must contain at least one value",
        ));
    }
    for lambda in &lambdas {
        if !lambda.is_finite() || *lambda < 0.0 {
            return Err(fitclinear_invalid(
                "fitclinear: Lambda values must be nonnegative finite scalars",
            ));
        }
    }
    options.lambdas = Some(lambdas);
    if let Some(beta0) = &options.beta0 {
        if beta0.len() != clean_cols {
            return Err(fitclinear_invalid(format!(
                "fitclinear: Beta must contain {} coefficients",
                clean_cols
            )));
        }
    }
    let priors = class_priors(&clean_classes, &clean_weights, options.prior.clone())?;
    Ok(FitSpec {
        x: clean_x,
        class_indices: clean_classes,
        class_names,
        class_kind: labels.kind,
        predictor_names,
        response_name,
        weights: clean_weights,
        priors,
        options,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments() -> FitclinearArguments {
        FitclinearArguments {
            option_start: 1,
            options: Vec::new(),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn matrix_admission_keeps_rank_one_and_rank_zero_observation_geometry() {
        let rank_one = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).expect("rank-one predictors");
        assert_eq!(classification_matrix_dimensions(&rank_one), (1, 3));
        let labels = Tensor::new(vec![0.0, 1.0, 0.0], vec![3, 1]).expect("three labels");
        let error = build_matrix_fit_spec(
            Value::Tensor(rank_one),
            Value::Tensor(labels),
            crate::call::FunctionArgumentView::from_slice(&[]),
            arguments(),
        )
        .expect_err("rank-one X is one observation, not three");
        assert!(error.message().contains("length(Y)"));

        let rank_zero = Tensor::new(vec![1.0], vec![]).expect("rank-zero predictors");
        assert_eq!(classification_matrix_dimensions(&rank_zero), (0, 0));
        let labels = Tensor::new(vec![0.0], vec![1, 1]).expect("one label");
        let error = build_matrix_fit_spec(
            Value::Tensor(rank_zero),
            Value::Tensor(labels),
            crate::call::FunctionArgumentView::from_slice(&[]),
            arguments(),
        )
        .expect_err("rank-zero X has no phantom observation");
        assert!(error.message().contains("length(Y)"));
    }
}
