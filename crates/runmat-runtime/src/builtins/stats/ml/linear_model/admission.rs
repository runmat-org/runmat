use super::*;

pub(super) fn build_fit_spec(
    first: Value,
    rest: crate::call::FunctionArgumentView<'_>,
    arguments: FitlmArguments,
) -> BuiltinResult<FitSpec> {
    if is_table_value(&first) {
        build_table_fit_spec(first, rest, arguments)
    } else {
        if rest.is_empty() {
            return Err(fitlm_invalid(
                "fitlm: matrix input requires response vector y",
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
    arguments: FitlmArguments,
) -> BuiltinResult<FitSpec> {
    let x = tensor::value_into_tensor_for(FITLM_NAME, x_value)
        .map_err(|err| fitlm_invalid(format!("fitlm: {err}")))?;
    ensure_exact_fitlm_integer_tensor_boundary(&x, "X")?;
    if x.shape().len() > 2 {
        return Err(fitlm_invalid("fitlm: X must be a 2-D numeric matrix"));
    }
    let y_tensor = tensor::value_into_tensor_for(FITLM_NAME, y_value)
        .map_err(|err| fitlm_invalid(format!("fitlm: {err}")))?;
    ensure_exact_fitlm_integer_tensor_boundary(&y_tensor, "y")?;
    let y = vector_values(&y_tensor, "y")?;
    let (rows, cols) = fitlm_matrix_dimensions(&x);
    if y.len() != rows {
        return Err(fitlm_invalid(
            "fitlm: length(y) must match the number of rows in X",
        ));
    }
    let mut options = FitOptions::default();
    let mut model_spec = ModelSpec::Linear;
    if arguments.option_start == 2 {
        model_spec = parse_model_spec(&rest[0], cols)?;
    }
    parse_fit_name_values(rest, arguments.options, 1, &mut options)?;
    if options.response_var.is_some() {
        return Err(fitlm_invalid(
            "fitlm: ResponseVar is only valid for table input",
        ));
    }
    if options.predictor_vars.is_some() {
        return Err(fitlm_invalid(
            "fitlm: PredictorVars is only valid for table input",
        ));
    }
    let (predictor_names, response_name) = if let Some(names) = options.var_names.clone() {
        if names.len() != cols + 1 {
            return Err(fitlm_invalid(format!(
                "fitlm: VarNames must contain {} predictor names and one response name",
                cols
            )));
        }
        (names[..cols].to_vec(), names[cols].clone())
    } else {
        (
            (1..=cols).map(|idx| format!("x{idx}")).collect(),
            "y".to_string(),
        )
    };
    let terms = model_terms(model_spec, cols, options.intercept);
    let coefficient_names = coefficient_names(&terms, &predictor_names);
    let model_formula = formula_text(&response_name, &coefficient_names);
    let weights = match options.weights {
        Some(WeightSpec::Values(values)) => Some(values),
        Some(WeightSpec::Variable(_)) => {
            return Err(fitlm_invalid(
                "fitlm: weights variable names require table input",
            ));
        }
        None => None,
    };
    Ok(FitSpec {
        x,
        y,
        predictor_names,
        response_name,
        terms,
        coefficient_names,
        model_formula,
        exclude: options.exclude,
        weights,
    })
}

fn build_table_fit_spec(
    first: Value,
    rest: crate::call::FunctionArgumentView<'_>,
    arguments: FitlmArguments,
) -> BuiltinResult<FitSpec> {
    let object = match first {
        Value::Object(object) => object,
        _ => unreachable!("table checked"),
    };
    let variable_names = table_variable_names_from_object(&object)
        .map_err(|err| fitlm_invalid(format!("fitlm: {err}")))?;
    if variable_names.len() < 2 {
        return Err(fitlm_invalid(
            "fitlm: table input must contain at least one predictor and one response",
        ));
    }
    let variables =
        table_variables(&object).map_err(|err| fitlm_invalid(format!("fitlm: {err}")))?;
    let mut options = FitOptions::default();
    let mut model_spec = ModelSpec::Linear;
    let mut formula: Option<String> = None;
    let mut response_arg: Option<String> = None;
    if arguments.option_start == 1 {
        if let Some(text) = scalar_text(&rest[0])? {
            if text.contains('~') {
                formula = Some(text.into_owned());
            } else if is_known_model_spec(&text) {
                model_spec = parse_model_spec_text(&text, variable_names.len() - 1)?;
            } else {
                response_arg = Some(text.into_owned());
            }
        } else {
            model_spec = parse_model_spec(&rest[0], variable_names.len() - 1)?;
        }
    }
    parse_fit_name_values(rest, arguments.options, 0, &mut options)?;
    let response_name = formula
        .as_ref()
        .and_then(|text| text.split_once('~').map(|(lhs, _)| lhs.trim().to_string()))
        .or(options.response_var.clone())
        .or(response_arg)
        .unwrap_or_else(|| variable_names.last().cloned().unwrap());
    if !variable_names.iter().any(|name| name == &response_name) {
        return Err(fitlm_invalid(format!(
            "fitlm: response variable '{response_name}' is not in the table"
        )));
    }
    let mut predictor_names = if let Some(text) = &formula {
        parse_formula_predictors(text, &variable_names, &response_name, &mut options)?
    } else if let Some(names) = options.predictor_vars.clone() {
        names
    } else {
        variable_names
            .iter()
            .filter(|name| *name != &response_name)
            .cloned()
            .collect()
    };
    let weights = match options.weights.clone() {
        Some(WeightSpec::Variable(name)) => {
            let value = variables.fields.get(&name).ok_or_else(|| {
                fitlm_invalid(format!(
                    "fitlm: weights variable '{name}' is not in the table"
                ))
            })?;
            predictor_names.retain(|predictor| predictor != &name);
            Some(numeric_vector(value, "Weights")?)
        }
        Some(WeightSpec::Values(values)) => Some(values),
        None => None,
    };
    if predictor_names.is_empty() && !options.intercept {
        return Err(fitlm_invalid("fitlm: model must contain at least one term"));
    }
    for name in &predictor_names {
        if !variable_names.iter().any(|candidate| candidate == name) {
            return Err(fitlm_invalid(format!(
                "fitlm: predictor variable '{name}' is not in the table"
            )));
        }
    }
    let y_value = variables
        .fields
        .get(&response_name)
        .ok_or_else(|| fitlm_invalid("fitlm: response variable missing from table"))?;
    let y_tensor = tensor::value_into_tensor_for(FITLM_NAME, y_value.clone())
        .map_err(|err| fitlm_invalid(format!("fitlm: {err}")))?;
    ensure_exact_fitlm_integer_tensor_boundary(&y_tensor, &response_name)?;
    let y = vector_values(&y_tensor, &response_name)?;
    let x = if predictor_names.is_empty() {
        Tensor::new(Vec::new(), vec![y.len(), 0])
            .map_err(|err| fitlm_internal(format!("fitlm: {err}")))?
    } else {
        let mut columns = Vec::with_capacity(predictor_names.len());
        for name in &predictor_names {
            let value = variables
                .fields
                .get(name)
                .ok_or_else(|| fitlm_invalid(format!("fitlm: predictor '{name}' missing")))?;
            let tensor = tensor::value_into_tensor_for(FITLM_NAME, value.clone())
                .map_err(|_| fitlm_invalid(format!("fitlm: predictor '{name}' must be numeric")))?;
            ensure_exact_fitlm_integer_tensor_boundary(&tensor, name)?;
            let values = vector_values(&tensor, name)?;
            if values.len() != y.len() {
                return Err(fitlm_invalid(format!(
                    "fitlm: predictor '{name}' height does not match response"
                )));
            }
            columns.push(values);
        }
        columns_to_tensor(columns)?
    };
    let terms = if let Some(text) = &formula {
        parse_formula_terms(text, &predictor_names, options.intercept)?
    } else {
        model_terms(model_spec, predictor_names.len(), options.intercept)
    };
    let coefficient_names = coefficient_names(&terms, &predictor_names);
    let model_formula = formula
        .as_ref()
        .cloned()
        .unwrap_or_else(|| formula_text(&response_name, &coefficient_names));
    Ok(FitSpec {
        x,
        y,
        predictor_names,
        response_name,
        terms,
        coefficient_names,
        model_formula,
        exclude: options.exclude,
        weights,
    })
}
