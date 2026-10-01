use super::*;

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
    values: &[Value],
    names: &[NamedOption],
    index_offset: usize,
    options: &mut FitOptions,
) -> BuiltinResult<()> {
    for named in names {
        let name = named.spelling.as_str();
        let value = &values[named.index - index_offset + 1];
        match named.identity {
            FitclinearOption::Learner => {
                let text = scalar_text(value)
                    .ok_or_else(|| fitclinear_invalid("fitclinear: Learner must be text"))?;
                options.learner = match text.to_ascii_lowercase().as_str() {
                    "svm" => Learner::Svm,
                    "logistic" => Learner::Logistic,
                    other => {
                        return Err(fitclinear_invalid(format!(
                            "fitclinear: unsupported Learner '{other}'"
                        )));
                    }
                };
            }
            FitclinearOption::Lambda => options.lambdas = Some(numeric_vector(value, "Lambda")?),
            FitclinearOption::Regularization => {
                let text = scalar_text(value)
                    .ok_or_else(|| fitclinear_invalid("fitclinear: Regularization must be text"))?;
                options.regularization = match text.to_ascii_lowercase().as_str() {
                    "ridge" => Regularization::Ridge,
                    "lasso" => Regularization::Lasso,
                    other => {
                        return Err(fitclinear_invalid(format!(
                            "fitclinear: unsupported Regularization '{other}'"
                        )));
                    }
                };
            }
            FitclinearOption::Solver => {
                let text = scalar_text(value)
                    .ok_or_else(|| fitclinear_invalid("fitclinear: Solver must be text"))?;
                options.solver = FitclinearSolver::from_public_name(&text).ok_or_else(|| {
                    fitclinear_invalid(format!(
                        "fitclinear: unsupported Solver '{}'",
                        text.to_ascii_lowercase()
                    ))
                })?;
            }
            FitclinearOption::ObservationsIn => {
                let text = scalar_text(value)
                    .ok_or_else(|| fitclinear_invalid("fitclinear: ObservationsIn must be text"))?;
                options.observations_in = match text.to_ascii_lowercase().as_str() {
                    "rows" => ObservationsIn::Rows,
                    "columns" => ObservationsIn::Columns,
                    other => {
                        return Err(fitclinear_invalid(format!(
                            "fitclinear: unsupported ObservationsIn '{other}'"
                        )));
                    }
                };
            }
            FitclinearOption::FitBias => options.fit_bias = logical_scalar(value, "FitBias")?,
            FitclinearOption::ClassNames => {
                let labels = label_vector(value.clone(), "ClassNames")?;
                options.class_names = Some(labels.labels);
            }
            FitclinearOption::PredictorNames => {
                options.predictor_names = Some(text_list(value, "PredictorNames")?);
            }
            FitclinearOption::ResponseName => {
                options.response_name =
                    Some(scalar_text(value).ok_or_else(|| {
                        fitclinear_invalid("fitclinear: ResponseName must be text")
                    })?);
            }
            FitclinearOption::Weights => {
                options.weights = Some(if let Some(name) = scalar_text(value) {
                    WeightSpec::Variable(name)
                } else {
                    WeightSpec::Values(numeric_vector(value, "Weights")?)
                });
            }
            FitclinearOption::Prior => {
                if let Some(text) = scalar_text(value) {
                    match text.to_ascii_lowercase().as_str() {
                        "empirical" => options.prior = None,
                        "uniform" => options.prior = Some(vec![0.5, 0.5]),
                        other => {
                            return Err(fitclinear_invalid(format!(
                                "fitclinear: unsupported Prior '{other}'"
                            )));
                        }
                    }
                } else {
                    options.prior = Some(numeric_vector(value, "Prior")?);
                }
            }
            FitclinearOption::ScoreTransform => {
                let text = scalar_text(value)
                    .ok_or_else(|| fitclinear_invalid("fitclinear: ScoreTransform must be text"))?;
                options.score_transform = Some(match text.to_ascii_lowercase().as_str() {
                    "none" => ScoreTransform::None,
                    "logit" => ScoreTransform::Logit,
                    other => {
                        return Err(fitclinear_invalid(format!(
                            "fitclinear: unsupported ScoreTransform '{other}'"
                        )));
                    }
                });
            }
            FitclinearOption::IterationLimit | FitclinearOption::PassLimit => {
                options.iteration_limit = positive_integer(value, &name)?;
            }
            FitclinearOption::Beta => options.beta0 = Some(numeric_vector(value, "Beta")?),
            FitclinearOption::Bias => options.bias0 = Some(numeric_scalar(value, "Bias")?),
            FitclinearOption::BatchLimit
            | FitclinearOption::BetaTolerance
            | FitclinearOption::DeltaGradientTolerance
            | FitclinearOption::GradientTolerance
            | FitclinearOption::NumCheckConvergence
            | FitclinearOption::PostFitBias
            | FitclinearOption::Verbose => {
                // Accepted for script compatibility. The CPU solver below is deterministic
                // batch gradient descent and records convergence diagnostics in FitInfo.
            }
            FitclinearOption::Cost
            | FitclinearOption::CrossVal
            | FitclinearOption::CvPartition
            | FitclinearOption::Holdout
            | FitclinearOption::KFold
            | FitclinearOption::Leaveout
            | FitclinearOption::OptimizeHyperparameters
            | FitclinearOption::HyperparameterOptimizationOptions => {
                return Err(fitclinear_invalid(format!(
                    "fitclinear: option '{name}' is not supported yet"
                )));
            }
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
