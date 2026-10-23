use super::*;

/// Projects dense predictors into classification-tree observation geometry.
///
/// A rank-one tensor is one observation row, while an empty-rank tensor is an
/// empty matrix. This domain policy deliberately differs from Tensor's general
/// matrix getter projection.
pub(super) fn classification_tree_matrix_dimensions(tensor: &Tensor) -> (usize, usize) {
    match tensor.shape() {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}

pub(super) fn build_fit_spec(
    first: Value,
    rest: crate::call::FunctionArgumentView<'_>,
    arguments: FitctreeArguments,
) -> BuiltinResult<FitSpec> {
    if is_table_value(&first) {
        build_table_fit_spec(first, rest, arguments)
    } else {
        if rest.is_empty() {
            return Err(fitctree_invalid(
                "fitctree: matrix input requires response vector Y",
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
    arguments: FitctreeArguments,
) -> BuiltinResult<FitSpec> {
    let x = tensor::value_into_tensor_for(FITCTREE_NAME, x_value)
        .map_err(|err| fitctree_invalid(format!("fitctree: {err}")))?;
    ensure_integer_tensor_exact(&x, "predictors")?;
    if x.shape().len() > 2 {
        return Err(fitctree_invalid("fitctree: X must be a 2-D numeric matrix"));
    }
    let (x_rows, x_cols) = classification_tree_matrix_dimensions(&x);
    let labels = label_vector(y_value, "Y")?;
    if labels.labels.len() != x_rows {
        return Err(fitctree_invalid(
            "fitctree: length(Y) must match the number of rows in X",
        ));
    }
    let mut options = FitOptions::default();
    parse_fit_options(rest, arguments.options, 1, &mut options)?;
    if let Some(WeightSpec::Variable(_)) = options.weights {
        return Err(fitctree_invalid(
            "fitctree: weights variable names require table input",
        ));
    }
    let predictor_names = if let Some(names) = options.predictor_names.clone() {
        if names.len() != x_cols {
            return Err(fitctree_invalid(format!(
                "fitctree: PredictorNames must contain {} names",
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
    arguments: FitctreeArguments,
) -> BuiltinResult<FitSpec> {
    let object = match first {
        Value::Object(object) => object,
        _ => unreachable!("table checked"),
    };
    let variable_names = table_variable_names_from_object(&object)
        .map_err(|err| fitctree_invalid(format!("fitctree: {err}")))?;
    if variable_names.len() < 2 {
        return Err(fitctree_invalid(
            "fitctree: table input must contain at least one predictor and one response",
        ));
    }
    let variables =
        table_variables(&object).map_err(|err| fitctree_invalid(format!("fitctree: {err}")))?;
    let mut options = FitOptions::default();
    let mut response_name = variable_names
        .last()
        .cloned()
        .ok_or_else(|| fitctree_invalid("fitctree: response variable missing"))?;
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
    if external_y.is_none() {
        response_name = options.response_name.clone().unwrap_or(response_name);
    }
    let has_external_y = external_y.is_some();
    let labels = if let Some(labels) = external_y {
        labels
    } else {
        let value = variables.fields.get(&response_name).ok_or_else(|| {
            fitctree_invalid(format!(
                "fitctree: response variable '{response_name}' is not in the table"
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
        return Err(fitctree_invalid(
            "fitctree: at least one predictor variable is required",
        ));
    }
    let mut columns = Vec::with_capacity(predictor_names.len());
    for name in &predictor_names {
        let value = variables.fields.get(name).ok_or_else(|| {
            fitctree_invalid(format!(
                "fitctree: predictor variable '{name}' is not in the table"
            ))
        })?;
        let tensor = tensor::value_into_tensor_for(FITCTREE_NAME, value.clone()).map_err(|_| {
            fitctree_invalid(format!(
                "fitctree: predictor variable '{name}' must be numeric"
            ))
        })?;
        ensure_integer_tensor_exact(&tensor, name)?;
        columns.push(vector_values(&tensor, name)?);
    }
    let x = columns_to_tensor(columns)?;
    let weights = match &options.weights {
        Some(WeightSpec::Values(values)) => values.clone(),
        Some(WeightSpec::Variable(name)) => {
            let value = variables.fields.get(name).ok_or_else(|| {
                fitctree_invalid(format!(
                    "fitctree: weights variable '{name}' is not in the table"
                ))
            })?;
            numeric_vector(value, "Weights")?
        }
        None => vec![1.0; classification_tree_matrix_dimensions(&x).0],
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
    let (x_rows, x_cols) = classification_tree_matrix_dimensions(&x);
    if labels.labels.len() != x_rows {
        return Err(fitctree_invalid(
            "fitctree: response length must match the number of observations",
        ));
    }
    if weights.len() != x_rows {
        return Err(fitctree_invalid(
            "fitctree: Weights length must match the number of observations",
        ));
    }
    for weight in &weights {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(fitctree_invalid(
                "fitctree: weights must be nonnegative finite values",
            ));
        }
    }
    let class_names = if let Some(names) = options.class_names.clone() {
        validate_class_names(names, labels.kind)?
    } else {
        unique_labels(&labels.labels, labels.kind)
    };
    if class_names.len() < 2 {
        return Err(fitctree_invalid(
            "fitctree: at least two classes are required",
        ));
    }
    let mut clean_rows = Vec::new();
    let mut clean_classes = Vec::new();
    let mut clean_weights = Vec::new();
    for row in 0..x_rows {
        let weight = weights[row];
        if weight == 0.0 {
            continue;
        }
        if is_missing_label(&labels.labels[row]) {
            continue;
        }
        let class_index = class_names
            .iter()
            .position(|label| same_label(label, &labels.labels[row]))
            .ok_or_else(|| {
                fitctree_invalid("fitctree: response contains a class outside ClassNames")
            })?;
        let mut row_values = Vec::with_capacity(x_cols);
        let mut has_observed_predictor = false;
        for col in 0..x_cols {
            let value = x_value(&x, row, col);
            if value.is_infinite() {
                return Err(fitctree_invalid(
                    "fitctree: predictor values must not contain Inf",
                ));
            }
            if !value.is_nan() {
                has_observed_predictor = true;
            }
            row_values.push(value);
        }
        if has_observed_predictor {
            clean_rows.push(row_values);
            clean_classes.push(class_index);
            clean_weights.push(weight);
        }
    }
    if clean_classes.is_empty() {
        return Err(fitctree_invalid(
            "fitctree: at least one weighted observation with an observed predictor is required",
        ));
    }
    let mut observed_classes = vec![false; class_names.len()];
    for class in &clean_classes {
        observed_classes[*class] = true;
    }
    if observed_classes
        .iter()
        .filter(|observed| **observed)
        .count()
        < 2
    {
        return Err(fitctree_invalid(
            "fitctree: at least two observed classes are required",
        ));
    }
    let mut clean_data = Vec::with_capacity(clean_rows.len() * x_cols);
    for col in 0..x_cols {
        for row in &clean_rows {
            clean_data.push(row[col]);
        }
    }
    let clean_x = Tensor::new(clean_data, vec![clean_rows.len(), x_cols])
        .map_err(|err| fitctree_internal(format!("fitctree: {err}")))?;
    if options.max_num_splits.is_none() {
        options.max_num_splits = Some(clean_classes.len().saturating_sub(1));
    }
    Ok(FitSpec {
        x: clean_x,
        class_indices: clean_classes,
        class_names,
        class_kind: labels.kind,
        predictor_names,
        response_name,
        weights: clean_weights,
        options,
    })
}
