use super::*;

pub(super) fn fit_models(
    runtime: Option<&crate::context::RuntimeContext>,
    spec: &FitSpec,
) -> BuiltinResult<FitResult> {
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
    let lambda = row_tensor(lambdas.clone())?;
    let learner = Value::string(spec.options.learner.as_str().to_string());
    let regularization = Value::string(spec.options.regularization.as_str().to_string());
    let solver = Value::String(spec.options.solver.public_name().into());
    let score_transform = Value::string(score_transform(spec).as_str().to_string());
    let response_name = Value::string(spec.response_name.clone());
    let predictor_names = Value::StringArray(
        StringArray::new(
            spec.predictor_names.clone(),
            vec![1, spec.predictor_names.len()],
        )
        .map_err(fitclinear_internal)?,
    );
    let class_names = labels_value(&spec.class_names, spec.class_kind)?;
    let prior = row_tensor(spec.priors.clone())?;
    let class_kind = Value::string(label_kind_name(spec.class_kind).to_string());
    let observations_in = Value::string(spec.options.observations_in.as_str().to_string());
    let fit_bias = Value::Bool(spec.options.fit_bias);
    let iteration_limit = Value::Num(spec.options.iteration_limit as f64);
    let parameter_fields = [
        ("Learner", &learner),
        ("Regularization", &regularization),
        ("Solver", &solver),
        ("ObservationsIn", &observations_in),
        ("FitBias", &fit_bias),
        ("IterationLimit", &iteration_limit),
    ];
    let parameters = crate::execution::construct_struct_from_fields_in_optional_context(
        runtime,
        parameter_fields.into_iter(),
    )?;
    let (parameters, _parameter_roots) = parameters.into_value_and_lease();
    let fields = [
        ("Beta", beta_value),
        ("Bias", bias_value),
        ("Lambda", lambda),
        ("Learner", learner),
        ("Regularization", regularization),
        ("Solver", solver),
        ("ScoreTransform", score_transform),
        ("FitBias", Value::Bool(spec.options.fit_bias)),
        ("ResponseName", response_name),
        ("PredictorNames", predictor_names),
        ("ClassNames", class_names),
        ("Prior", prior),
        ("NumObservations", Value::Num(observations as f64)),
        ("NumPredictors", Value::Num(predictors as f64)),
        ("__RunMatLinearClassKind", class_kind),
        ("ModelParameters", parameters),
    ];
    let object = crate::execution::object::construct_object_from_borrowed_fields(
        runtime,
        runmat_types::standard::CLASSIFICATION_LINEAR.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
    )?;
    let (object, _object_roots) = object.into_value_and_lease();
    let Value::Object(object) = object else {
        unreachable!("object constructor publishes Object")
    };
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
