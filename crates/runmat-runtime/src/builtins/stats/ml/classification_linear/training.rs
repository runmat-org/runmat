use super::*;

pub(super) fn fit_models(spec: &FitSpec) -> BuiltinResult<FitResult> {
    let lambdas = spec
        .options
        .lambdas
        .as_ref()
        .expect("lambdas populated during finalize");
    let mut models = Vec::with_capacity(lambdas.len());
    for lambda in lambdas {
        models.push(train_for_lambda(spec, *lambda)?);
    }
    let (observations, predictors) = classification_matrix_dimensions(&spec.x);
    let beta_value = beta_value(&models, predictors)?;
    let bias_value = row_tensor(models.iter().map(|model| model.bias).collect())?;
    let mut object = ObjectInstance::new(runmat_types::standard::CLASSIFICATION_LINEAR.to_string());
    object
        .properties
        .insert("Beta".to_string(), beta_value.clone());
    object
        .properties
        .insert("Bias".to_string(), bias_value.clone());
    object
        .properties
        .insert("Lambda".to_string(), row_tensor(lambdas.clone())?);
    object.properties.insert(
        "Learner".to_string(),
        Value::String(spec.options.learner.as_str().to_string()),
    );
    object.properties.insert(
        "Regularization".to_string(),
        Value::String(spec.options.regularization.as_str().to_string()),
    );
    object.properties.insert(
        "Solver".to_string(),
        Value::String(spec.options.solver.public_name().into()),
    );
    object.properties.insert(
        "ScoreTransform".to_string(),
        Value::String(score_transform(spec).as_str().to_string()),
    );
    object
        .properties
        .insert("FitBias".to_string(), Value::Bool(spec.options.fit_bias));
    object.properties.insert(
        "ResponseName".to_string(),
        Value::String(spec.response_name.clone()),
    );
    object.properties.insert(
        "PredictorNames".to_string(),
        Value::StringArray(StringArray {
            data: spec.predictor_names.clone(),
            shape: vec![1, spec.predictor_names.len()],
            rows: 1,
            cols: spec.predictor_names.len(),
        }),
    );
    object.properties.insert(
        "ClassNames".to_string(),
        labels_value(&spec.class_names, spec.class_kind)?,
    );
    object
        .properties
        .insert("Prior".to_string(), row_tensor(spec.priors.clone())?);
    object.properties.insert(
        "NumObservations".to_string(),
        Value::Num(observations as f64),
    );
    object
        .properties
        .insert("NumPredictors".to_string(), Value::Num(predictors as f64));
    object.properties.insert(
        "__RunMatLinearClassKind".to_string(),
        Value::String(label_kind_name(spec.class_kind).to_string()),
    );
    let mut parameters = StructValue::new();
    parameters.insert(
        "Learner",
        Value::String(spec.options.learner.as_str().to_string()),
    );
    parameters.insert(
        "Regularization",
        Value::String(spec.options.regularization.as_str().to_string()),
    );
    parameters.insert(
        "Solver",
        Value::String(spec.options.solver.public_name().into()),
    );
    parameters.insert(
        "ObservationsIn",
        Value::String(spec.options.observations_in.as_str().to_string()),
    );
    parameters.insert("FitBias", Value::Bool(spec.options.fit_bias));
    parameters.insert(
        "IterationLimit",
        Value::Num(spec.options.iteration_limit as f64),
    );
    object
        .properties
        .insert("ModelParameters".to_string(), Value::Struct(parameters));
    let fit_info = fit_info_value(spec, lambdas, &models)?;
    Ok(FitResult { object, fit_info })
}

fn train_for_lambda(spec: &FitSpec, lambda: f64) -> BuiltinResult<TrainedModel> {
    let (observations, predictors) = classification_matrix_dimensions(&spec.x);
    let mut beta = spec
        .options
        .beta0
        .clone()
        .unwrap_or_else(|| vec![0.0; predictors]);
    let mut bias = spec.options.bias0.unwrap_or_else(|| {
        if spec.options.fit_bias {
            (spec.priors[1] / spec.priors[0]).ln() / 2.0
        } else {
            0.0
        }
    });
    let total_weight = spec.weights.iter().sum::<f64>().max(EPS);
    let base_step = if spec.options.learner == Learner::Logistic {
        0.25
    } else {
        0.15
    };
    let mut last_grad_norm = f64::INFINITY;
    for iter in 0..spec.options.iteration_limit.max(1) {
        let mut grad_beta = vec![0.0; predictors];
        let mut grad_bias = 0.0;
        for row in 0..observations {
            let y = if spec.class_indices[row] == 1 {
                1.0
            } else {
                -1.0
            };
            let margin = dot_row(&spec.x, row, &beta) + bias;
            let coeff = match spec.options.learner {
                Learner::Logistic => {
                    let ym = (y * margin).clamp(-40.0, 40.0);
                    -y / (1.0 + ym.exp())
                }
                Learner::Svm => {
                    if y * margin < 1.0 {
                        -y
                    } else {
                        0.0
                    }
                }
            } * spec.weights[row]
                / total_weight;
            if coeff != 0.0 {
                for (col, grad) in grad_beta.iter_mut().enumerate() {
                    *grad += coeff * x_value(&spec.x, row, col);
                }
                if spec.options.fit_bias {
                    grad_bias += coeff;
                }
            }
        }
        match spec.options.regularization {
            Regularization::Ridge => {
                for (grad, coef) in grad_beta.iter_mut().zip(beta.iter()) {
                    *grad += lambda * *coef;
                }
            }
            Regularization::Lasso => {}
        }
        last_grad_norm = grad_beta
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
            .hypot(grad_bias.abs());
        let step = base_step / (1.0 + lambda + 0.01 * iter as f64).sqrt();
        for (coef, grad) in beta.iter_mut().zip(grad_beta.iter()) {
            *coef -= step * *grad;
            if spec.options.regularization == Regularization::Lasso && lambda > 0.0 {
                *coef = soft_threshold(*coef, step * lambda);
            }
        }
        if spec.options.fit_bias {
            bias -= step * grad_bias;
        } else {
            bias = 0.0;
        }
        if last_grad_norm < 1.0e-7 {
            break;
        }
    }
    Ok(TrainedModel {
        objective: objective(spec, &beta, bias, lambda),
        beta,
        bias,
        iterations: spec.options.iteration_limit.max(1),
        gradient_norm: last_grad_norm,
    })
}

fn objective(spec: &FitSpec, beta: &[f64], bias: f64, lambda: f64) -> f64 {
    let total_weight = spec.weights.iter().sum::<f64>().max(EPS);
    let mut loss = 0.0;
    let observations = classification_matrix_dimensions(&spec.x).0;
    for row in 0..observations {
        let y = if spec.class_indices[row] == 1 {
            1.0
        } else {
            -1.0
        };
        let margin = dot_row(&spec.x, row, beta) + bias;
        let row_loss = match spec.options.learner {
            Learner::Logistic => (1.0 + (-y * margin).clamp(-40.0, 40.0).exp()).ln(),
            Learner::Svm => (1.0 - y * margin).max(0.0),
        };
        loss += spec.weights[row] * row_loss / total_weight;
    }
    let penalty = match spec.options.regularization {
        Regularization::Ridge => 0.5 * lambda * beta.iter().map(|value| value * value).sum::<f64>(),
        Regularization::Lasso => lambda * beta.iter().map(|value| value.abs()).sum::<f64>(),
    };
    loss + penalty
}
