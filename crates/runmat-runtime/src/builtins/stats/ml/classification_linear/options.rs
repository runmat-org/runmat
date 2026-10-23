use super::*;

pub(super) enum FitclinearAction {
    Learner(Learner),
    Lambda(Vec<f64>),
    Regularization(Regularization),
    Solver(FitclinearSolver),
    ObservationsIn(ObservationsIn),
    FitBias(bool),
    ClassNames(Vec<ClassLabel>),
    PredictorNames(Vec<String>),
    ResponseName(String),
    Weights(WeightSpec),
    Prior(Option<Vec<f64>>),
    ScoreTransform(ScoreTransform),
    IterationLimit(usize),
    Beta(Vec<f64>),
    Bias(f64),
    Ignored,
}

pub(super) fn decode_fit_option(
    named: &NamedOption,
    value: &Value,
) -> BuiltinResult<FitclinearAction> {
    Ok(match named.identity {
        FitclinearOption::Learner => {
            let text = scalar_text(value)?
                .ok_or_else(|| fitclinear_invalid("fitclinear: Learner must be text"))?;
            FitclinearAction::Learner(match text.to_ascii_lowercase().as_str() {
                "svm" => Learner::Svm,
                "logistic" => Learner::Logistic,
                other => {
                    return Err(fitclinear_invalid(format!(
                        "fitclinear: unsupported Learner '{other}'"
                    )))
                }
            })
        }
        FitclinearOption::Lambda => FitclinearAction::Lambda(numeric_vector(value, "Lambda")?),
        FitclinearOption::Regularization => {
            let text = scalar_text(value)?
                .ok_or_else(|| fitclinear_invalid("fitclinear: Regularization must be text"))?;
            FitclinearAction::Regularization(match text.to_ascii_lowercase().as_str() {
                "ridge" => Regularization::Ridge,
                "lasso" => Regularization::Lasso,
                other => {
                    return Err(fitclinear_invalid(format!(
                        "fitclinear: unsupported Regularization '{other}'"
                    )))
                }
            })
        }
        FitclinearOption::Solver => {
            let text = scalar_text(value)?
                .ok_or_else(|| fitclinear_invalid("fitclinear: Solver must be text"))?;
            FitclinearAction::Solver(FitclinearSolver::from_public_name(&text).ok_or_else(
                || {
                    fitclinear_invalid(format!(
                        "fitclinear: unsupported Solver '{}'",
                        text.to_ascii_lowercase()
                    ))
                },
            )?)
        }
        FitclinearOption::ObservationsIn => {
            let text = scalar_text(value)?
                .ok_or_else(|| fitclinear_invalid("fitclinear: ObservationsIn must be text"))?;
            FitclinearAction::ObservationsIn(match text.to_ascii_lowercase().as_str() {
                "rows" => ObservationsIn::Rows,
                "columns" => ObservationsIn::Columns,
                other => {
                    return Err(fitclinear_invalid(format!(
                        "fitclinear: unsupported ObservationsIn '{other}'"
                    )))
                }
            })
        }
        FitclinearOption::FitBias => FitclinearAction::FitBias(logical_scalar(value, "FitBias")?),
        FitclinearOption::ClassNames => {
            FitclinearAction::ClassNames(label_vector(value.clone(), "ClassNames")?.labels)
        }
        FitclinearOption::PredictorNames => {
            FitclinearAction::PredictorNames(text_list(value, "PredictorNames")?)
        }
        FitclinearOption::ResponseName => FitclinearAction::ResponseName(
            scalar_text(value)?
                .ok_or_else(|| fitclinear_invalid("fitclinear: ResponseName must be text"))?,
        ),
        FitclinearOption::Weights => {
            FitclinearAction::Weights(if let Some(name) = scalar_text(value)? {
                WeightSpec::Variable(name)
            } else {
                WeightSpec::Values(numeric_vector(value, "Weights")?)
            })
        }
        FitclinearOption::Prior => {
            let prior = if let Some(text) = scalar_text(value)? {
                match text.to_ascii_lowercase().as_str() {
                    "empirical" => None,
                    "uniform" => Some(vec![0.5, 0.5]),
                    other => {
                        return Err(fitclinear_invalid(format!(
                            "fitclinear: unsupported Prior '{other}'"
                        )))
                    }
                }
            } else {
                Some(numeric_vector(value, "Prior")?)
            };
            FitclinearAction::Prior(prior)
        }
        FitclinearOption::ScoreTransform => {
            let text = scalar_text(value)?
                .ok_or_else(|| fitclinear_invalid("fitclinear: ScoreTransform must be text"))?;
            FitclinearAction::ScoreTransform(match text.to_ascii_lowercase().as_str() {
                "none" => ScoreTransform::None,
                "logit" => ScoreTransform::Logit,
                other => {
                    return Err(fitclinear_invalid(format!(
                        "fitclinear: unsupported ScoreTransform '{other}'"
                    )))
                }
            })
        }
        FitclinearOption::IterationLimit | FitclinearOption::PassLimit => {
            FitclinearAction::IterationLimit(positive_integer(value, &named.spelling)?)
        }
        FitclinearOption::Beta => FitclinearAction::Beta(numeric_vector(value, "Beta")?),
        FitclinearOption::Bias => FitclinearAction::Bias(numeric_scalar(value, "Bias")?),
        FitclinearOption::BatchLimit
        | FitclinearOption::BetaTolerance
        | FitclinearOption::DeltaGradientTolerance
        | FitclinearOption::GradientTolerance
        | FitclinearOption::NumCheckConvergence
        | FitclinearOption::PostFitBias
        | FitclinearOption::Verbose => FitclinearAction::Ignored,
        FitclinearOption::Cost
        | FitclinearOption::CrossVal
        | FitclinearOption::CvPartition
        | FitclinearOption::Holdout
        | FitclinearOption::KFold
        | FitclinearOption::Leaveout
        | FitclinearOption::OptimizeHyperparameters
        | FitclinearOption::HyperparameterOptimizationOptions => {
            return Err(fitclinear_invalid(format!(
                "fitclinear: option '{}' is not supported yet",
                named.spelling
            )));
        }
    })
}

pub(super) fn class_priors(
    class_indices: &[usize],
    weights: &[f64],
    explicit: Option<Vec<f64>>,
) -> BuiltinResult<Vec<f64>> {
    if let Some(prior) = explicit {
        if prior.len() != 2 {
            return Err(fitclinear_invalid(
                "fitclinear: Prior must contain two class probabilities",
            ));
        }
        if prior.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return Err(fitclinear_invalid(
                "fitclinear: Prior values must be nonnegative finite values",
            ));
        }
        let sum = prior.iter().sum::<f64>();
        if sum <= 0.0 {
            return Err(fitclinear_invalid(
                "fitclinear: Prior values must have positive total",
            ));
        }
        return Ok(prior.into_iter().map(|value| value / sum).collect());
    }
    let mut counts = [0.0, 0.0];
    for (idx, weight) in class_indices.iter().zip(weights.iter()) {
        counts[*idx] += *weight;
    }
    let total = counts[0] + counts[1];
    Ok(vec![counts[0] / total, counts[1] / total])
}

pub(super) fn parse_fit_options(
    values: crate::call::FunctionArgumentView<'_>,
    names: Vec<PreparedOption>,
    index_offset: usize,
    options: &mut FitOptions,
) -> BuiltinResult<()> {
    for named in names {
        let action = match named.predecoded {
            Some(action) => action,
            None => decode_fit_option(&named.name, &values[named.name.index - index_offset + 1])?,
        };
        match action {
            FitclinearAction::Learner(value) => options.learner = value,
            FitclinearAction::Lambda(value) => options.lambdas = Some(value),
            FitclinearAction::Regularization(value) => options.regularization = value,
            FitclinearAction::Solver(value) => options.solver = value,
            FitclinearAction::ObservationsIn(value) => options.observations_in = value,
            FitclinearAction::FitBias(value) => options.fit_bias = value,
            FitclinearAction::ClassNames(value) => options.class_names = Some(value),
            FitclinearAction::PredictorNames(value) => options.predictor_names = Some(value),
            FitclinearAction::ResponseName(value) => options.response_name = Some(value),
            FitclinearAction::Weights(value) => options.weights = Some(value),
            FitclinearAction::Prior(value) => options.prior = value,
            FitclinearAction::ScoreTransform(value) => options.score_transform = Some(value),
            FitclinearAction::IterationLimit(value) => options.iteration_limit = value,
            FitclinearAction::Beta(value) => options.beta0 = Some(value),
            FitclinearAction::Bias(value) => options.bias0 = Some(value),
            FitclinearAction::Ignored => {}
        }
    }
    Ok(())
}

pub(super) fn parse_formula(
    text: &str,
    variables: &[String],
) -> BuiltinResult<(String, Vec<String>)> {
    let Some((lhs, rhs)) = text.split_once('~') else {
        return Err(fitclinear_invalid("fitclinear: formula must contain '~'"));
    };
    let response = lhs.trim().to_string();
    if response.is_empty() {
        return Err(fitclinear_invalid("fitclinear: formula response is empty"));
    }
    let mut predictors = Vec::new();
    for term in rhs.split('+') {
        let term = term.trim();
        if term.is_empty() || term == "1" {
            continue;
        }
        if term.contains('*') || term.contains(':') || term.contains('^') {
            return Err(fitclinear_invalid(
                "fitclinear: formula interactions are not supported yet",
            ));
        }
        if !variables.iter().any(|name| name == term) {
            return Err(fitclinear_invalid(format!(
                "fitclinear: formula predictor '{term}' is not in the table"
            )));
        }
        predictors.push(term.to_string());
    }
    if predictors.is_empty() {
        return Err(fitclinear_invalid(
            "fitclinear: formula must contain at least one predictor",
        ));
    }
    Ok((response, predictors))
}
