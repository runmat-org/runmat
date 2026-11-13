use super::*;

pub(super) fn predict_linear_model(
    model: Value,
    xnew: Value,
    options: PredictOptions,
) -> BuiltinResult<(Value, Value)> {
    let object = match model {
        Value::Object(object) if object.class_name().is(runmat_types::standard::LINEAR_MODEL) => {
            object
        }
        other => {
            return Err(predict_invalid(format!(
                "predict: expected LinearModel object, got {other:?}"
            )));
        }
    };
    let terms = terms_from_object(&object)?;
    let beta = numeric_property(&object, "__RunMatBeta")?;
    let covariance = matrix_property(&object, "__RunMatCovariance")?;
    let predictor_names = string_property(&object, "PredictorNames")?;
    let x = predictors_for_prediction(xnew, &predictor_names)?;
    let design = design_matrix(&x, &terms);
    let rows = fitlm_matrix_dimensions(&x).0;
    let cols = terms.len();
    if beta.len() != cols {
        return Err(predict_invalid(
            "predict: model coefficient length is invalid",
        ));
    }
    let mut ypred = Vec::with_capacity(rows);
    let mut ci_low = Vec::with_capacity(rows);
    let mut ci_high = Vec::with_capacity(rows);
    let mse = numeric_scalar_property(&object, "__RunMatMSE")?;
    let dfe = numeric_scalar_property(&object, "__RunMatDFE")?;
    let crit = if dfe > 0.0 {
        student_t_inv(1.0 - options.alpha / 2.0, dfe)
    } else {
        f64::NAN
    };
    for row in 0..rows {
        let mut y = 0.0;
        for col in 0..cols {
            y += design[row * cols + col] * beta[col];
        }
        ypred.push(y);
        let leverage = leverage_for_row(&design[row * cols..row * cols + cols], &covariance, cols);
        let variance = match options.prediction {
            PredictionKind::Curve => leverage,
            PredictionKind::Observation => leverage + mse,
        };
        let delta = if variance >= 0.0 {
            crit * variance.sqrt()
        } else {
            f64::NAN
        };
        ci_low.push(y - delta);
        ci_high.push(y + delta);
    }
    let mut intervals = ci_low;
    intervals.extend(ci_high);
    let y_value = Value::Tensor(
        Tensor::new(ypred, vec![rows, 1])
            .map_err(|err| predict_internal(format!("predict: {err}")))?,
    );
    let ci_value = Value::Tensor(
        Tensor::new(intervals, vec![rows, 2])
            .map_err(|err| predict_internal(format!("predict: {err}")))?,
    );
    Ok((y_value, ci_value))
}

fn predictors_for_prediction(value: Value, predictor_names: &[String]) -> BuiltinResult<Tensor> {
    if let Value::Object(object) = &value {
        if is_tabular_object(object) {
            if predictor_names.is_empty() {
                let rows = table_height(object)
                    .map_err(|err| predict_invalid(format!("predict: {err}")))?;
                return Tensor::new(Vec::new(), vec![rows, 0])
                    .map_err(|err| predict_internal(format!("predict: {err}")));
            }
            let variables = table_variables(object)
                .map_err(|err| predict_invalid(format!("predict: {err}")))?;
            let mut columns = Vec::with_capacity(predictor_names.len());
            let mut height = None;
            for name in predictor_names {
                let raw = variables.fields.get(name).ok_or_else(|| {
                    predict_invalid(format!("predict: missing predictor '{name}'"))
                })?;
                let tensor =
                    tensor::value_into_tensor_for(PREDICT_NAME, raw.clone()).map_err(|_| {
                        predict_invalid(format!("predict: predictor '{name}' must be numeric"))
                    })?;
                let values = vector_values_predict(&tensor, name)?;
                if let Some(expected) = height {
                    if values.len() != expected {
                        return Err(predict_invalid(
                            "predict: predictor table variables must have the same height",
                        ));
                    }
                } else {
                    height = Some(values.len());
                }
                columns.push(values);
            }
            return columns_to_tensor_predict(columns);
        }
    }
    let tensor = tensor::value_into_tensor_for(PREDICT_NAME, value)
        .map_err(|err| predict_invalid(format!("predict: {err}")))?;
    if tensor.shape().len() > 2 || fitlm_matrix_dimensions(&tensor).1 != predictor_names.len() {
        return Err(predict_invalid(format!(
            "predict: Xnew must have {} predictor columns",
            predictor_names.len()
        )));
    }
    Ok(tensor)
}

fn design_matrix(x: &Tensor, terms: &[Vec<u32>]) -> Vec<f64> {
    let (rows, cols) = fitlm_matrix_dimensions(x);
    let mut out = Vec::with_capacity(rows * terms.len());
    for row in 0..rows {
        let raw = (0..cols)
            .map(|col| x_value(x, row, col))
            .collect::<Vec<_>>();
        for term in terms {
            out.push(evaluate_term(&raw, term));
        }
    }
    out
}

fn leverage_for_row(row: &[f64], covariance: &[f64], cols: usize) -> f64 {
    let mut value = 0.0;
    for i in 0..cols {
        for j in 0..cols {
            value += row[i] * covariance[i + j * cols] * row[j];
        }
    }
    value
}
