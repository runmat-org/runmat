use super::*;

pub(crate) fn predict_classification_linear_object(
    object: ObjectInstance,
    xnew: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Vec<Value>> {
    let options = parse_predict_options(&rest)?;
    let predictor_names = string_array_property(&object, "PredictorNames")?;
    let x = predictors_for_prediction(xnew, &predictor_names, options.observations_in)?;
    let beta = tensor_property(&object, "Beta")?;
    if beta.shape.len() > 2 || beta.rows != predictor_names.len() {
        return Err(predict_invalid("predict: model Beta has invalid shape"));
    }
    let bias = numeric_vector_property(&object, "Bias")?;
    if bias.len() != beta.cols {
        return Err(predict_invalid(
            "predict: model Bias length does not match Beta",
        ));
    }
    let class_names = labels_from_value(
        object
            .properties
            .get("ClassNames")
            .cloned()
            .ok_or_else(|| predict_invalid("predict: model is missing ClassNames"))?,
        "ClassNames",
    )?;
    if class_names.labels.len() != 2 {
        return Err(predict_invalid(
            "predict: model must contain two class names",
        ));
    }
    let score_transform = score_transform_property(&object)?;
    let mut label_indices = Vec::with_capacity(x.rows * beta.cols);
    let mut score_data = Vec::with_capacity(x.rows * 2 * beta.cols);
    let beta_values = beta.materialize_f64();
    for model_idx in 0..beta.cols {
        let model_beta = (0..beta.rows)
            .map(|row| beta_values[row + model_idx * beta.rows])
            .collect::<Vec<_>>();
        for row in 0..x.rows {
            let margin = dot_row(&x, row, &model_beta) + bias[model_idx];
            label_indices.push(if margin >= 0.0 { 1 } else { 0 });
        }
        for class_col in 0..2 {
            for row in 0..x.rows {
                let margin = dot_row(&x, row, &model_beta) + bias[model_idx];
                let value = match score_transform {
                    ScoreTransform::Logit => {
                        let p = sigmoid(margin);
                        if class_col == 0 {
                            1.0 - p
                        } else {
                            p
                        }
                    }
                    ScoreTransform::None => {
                        if class_col == 0 {
                            -margin
                        } else {
                            margin
                        }
                    }
                };
                score_data.push(value);
            }
        }
    }
    let label = predicted_labels_value(
        &class_names.labels,
        class_names.kind,
        &label_indices,
        x.rows,
        beta.cols,
    )?;
    let score_shape = if beta.cols == 1 {
        vec![x.rows, 2]
    } else {
        vec![x.rows, 2, beta.cols]
    };
    let score = Value::Tensor(
        Tensor::new(score_data, score_shape)
            .map_err(|err| predict_internal(format!("predict: {err}")))?,
    );
    Ok(vec![label, score])
}

#[derive(Clone, Copy, Debug)]
struct PredictOptions {
    observations_in: ObservationsIn,
}

fn parse_predict_options(values: &[Value]) -> BuiltinResult<PredictOptions> {
    if !values.len().is_multiple_of(2) {
        return Err(predict_invalid(
            "predict: name-value options must be paired",
        ));
    }
    let mut options = PredictOptions {
        observations_in: ObservationsIn::Rows,
    };
    let mut index = 0usize;
    while index < values.len() {
        let name = scalar_text(&values[index])
            .ok_or_else(|| predict_invalid("predict: option name must be text"))?;
        let value = &values[index + 1];
        match name.to_ascii_lowercase().as_str() {
            "observationsin" => {
                let text = scalar_text(value)
                    .ok_or_else(|| predict_invalid("predict: ObservationsIn must be text"))?;
                options.observations_in = match text.to_ascii_lowercase().as_str() {
                    "rows" => ObservationsIn::Rows,
                    "columns" => ObservationsIn::Columns,
                    other => {
                        return Err(predict_invalid(format!(
                            "predict: unsupported ObservationsIn '{other}'"
                        )));
                    }
                };
            }
            other => {
                return Err(predict_invalid(format!(
                    "predict: unknown option '{other}'"
                )));
            }
        }
        index += 2;
    }
    Ok(options)
}

fn predictors_for_prediction(
    value: Value,
    predictor_names: &[String],
    observations_in: ObservationsIn,
) -> BuiltinResult<Tensor> {
    if let Value::Object(object) = &value {
        if is_tabular_object(object) {
            if observations_in == ObservationsIn::Columns {
                return Err(predict_invalid(
                    "predict: ObservationsIn='columns' is not supported for table input",
                ));
            }
            let variables = table_variables(object)
                .map_err(|err| predict_invalid(format!("predict: {err}")))?;
            let mut columns = Vec::with_capacity(predictor_names.len());
            for name in predictor_names {
                let raw = variables.fields.get(name).ok_or_else(|| {
                    predict_invalid(format!("predict: missing predictor '{name}'"))
                })?;
                let tensor = numeric_matrix_for_predict(raw.clone()).map_err(|_| {
                    predict_invalid(format!("predict: predictor '{name}' must be numeric"))
                })?;
                columns.push(vector_values_predict(&tensor, name)?);
            }
            return columns_to_tensor_predict(columns);
        }
    }
    let tensor = normalize_prediction_matrix(numeric_matrix_for_predict(value)?, observations_in)?;
    if tensor.shape.len() > 2 || tensor.cols != predictor_names.len() {
        return Err(predict_invalid(format!(
            "predict: X must have {} predictor columns",
            predictor_names.len()
        )));
    }
    Ok(tensor)
}
