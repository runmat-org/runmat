use super::*;

pub(super) fn parse_fit_options(values: &[Value], options: &mut FitOptions) -> BuiltinResult<()> {
    if !values.len().is_multiple_of(2) {
        return Err(fitctree_invalid(
            "fitctree: name-value options must be paired",
        ));
    }
    let mut index = 0usize;
    while index < values.len() {
        let name = scalar_text(&values[index])
            .ok_or_else(|| fitctree_invalid("fitctree: option name must be text"))?;
        let value = &values[index + 1];
        match FitctreeOption::from_public_name(&name) {
            Some(FitctreeOption::ClassNames) => {
                let labels = label_vector(value.clone(), "ClassNames")?;
                options.class_names = Some(labels.labels);
            }
            Some(FitctreeOption::PredictorNames) => {
                options.predictor_names = Some(text_list(value, "PredictorNames")?);
            }
            Some(FitctreeOption::ResponseName) => {
                options.response_name = Some(
                    scalar_text(value)
                        .ok_or_else(|| fitctree_invalid("fitctree: ResponseName must be text"))?,
                );
            }
            Some(FitctreeOption::MaxNumSplits) => {
                options.max_num_splits = Some(nonnegative_integer(value, "MaxNumSplits")?);
            }
            Some(FitctreeOption::MinLeafSize) => {
                options.min_leaf_size = positive_integer(value, "MinLeafSize")?;
            }
            Some(FitctreeOption::MinParentSize) => {
                options.min_parent_size = positive_integer(value, "MinParentSize")?;
            }
            Some(FitctreeOption::SplitCriterion) => {
                let text = scalar_text(value)
                    .ok_or_else(|| fitctree_invalid("fitctree: SplitCriterion must be text"))?;
                options.split_criterion = match text.to_ascii_lowercase().as_str() {
                    "gdi" => SplitCriterion::Gdi,
                    "deviance" => SplitCriterion::Deviance,
                    other => {
                        return Err(fitctree_invalid(format!(
                            "fitctree: unsupported SplitCriterion '{other}'"
                        )));
                    }
                };
            }
            Some(FitctreeOption::Weights) => {
                options.weights = Some(if let Some(name) = scalar_text(value) {
                    WeightSpec::Variable(name)
                } else {
                    WeightSpec::Values(numeric_vector(value, "Weights")?)
                });
            }
            Some(FitctreeOption::ScoreTransform) => {
                let text = scalar_text(value)
                    .ok_or_else(|| fitctree_invalid("fitctree: ScoreTransform must be text"))?;
                if !text.eq_ignore_ascii_case("none") {
                    return Err(fitctree_invalid(
                        "fitctree: only ScoreTransform='none' is supported",
                    ));
                }
            }
            Some(
                FitctreeOption::AlgorithmForCategorical
                | FitctreeOption::CategoricalPredictors
                | FitctreeOption::Cost
                | FitctreeOption::CrossVal
                | FitctreeOption::CvPartition
                | FitctreeOption::Holdout
                | FitctreeOption::KFold
                | FitctreeOption::Leaveout
                | FitctreeOption::MaxNumCategories
                | FitctreeOption::MergeLeaves
                | FitctreeOption::NumBins
                | FitctreeOption::NumVariablesToSample
                | FitctreeOption::OptimizeHyperparameters
                | FitctreeOption::HyperparameterOptimizationOptions
                | FitctreeOption::PredictorSelection
                | FitctreeOption::Prior
                | FitctreeOption::Prune
                | FitctreeOption::PruneCriterion
                | FitctreeOption::Reproducible
                | FitctreeOption::Surrogate,
            ) => {
                return Err(fitctree_invalid(format!(
                    "fitctree: option '{name}' is not supported yet"
                )));
            }
            None => {
                return Err(fitctree_invalid(format!(
                    "fitctree: unknown option '{}'",
                    name.to_ascii_lowercase()
                )))
            }
        }
        index += 2;
    }
    Ok(())
}

pub(super) fn parse_formula(
    text: &str,
    variables: &[String],
) -> BuiltinResult<(String, Vec<String>)> {
    let Some((lhs, rhs)) = text.split_once('~') else {
        return Err(fitctree_invalid("fitctree: formula must contain '~'"));
    };
    let response = lhs.trim().to_string();
    if response.is_empty() {
        return Err(fitctree_invalid("fitctree: formula response is empty"));
    }
    let mut predictors = Vec::new();
    for term in rhs.split('+') {
        let term = term.trim();
        if term.is_empty() {
            continue;
        }
        if term == "1" {
            continue;
        }
        if term.contains('*') || term.contains(':') || term.contains('^') {
            return Err(fitctree_invalid(
                "fitctree: formula interactions are not supported yet",
            ));
        }
        if !variables.iter().any(|name| name == term) {
            return Err(fitctree_invalid(format!(
                "fitctree: formula predictor '{term}' is not in the table"
            )));
        }
        predictors.push(term.to_string());
    }
    if predictors.is_empty() {
        return Err(fitctree_invalid(
            "fitctree: formula must contain at least one predictor",
        ));
    }
    Ok((response, predictors))
}
