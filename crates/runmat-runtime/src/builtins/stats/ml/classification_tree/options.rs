use super::header::{NamedOption, PreparedOption};
use super::*;

pub(super) enum FitctreeAction {
    ClassNames(Vec<ClassLabel>),
    PredictorNames(Vec<String>),
    ResponseName(String),
    MaxNumSplits(usize),
    MinLeafSize(usize),
    MinParentSize(usize),
    SplitCriterion(SplitCriterion),
    Weights(WeightSpec),
    ScoreTransformNone,
}

pub(super) fn decode_option(name: &NamedOption, value: &Value) -> BuiltinResult<FitctreeAction> {
    Ok(match name.identity {
        FitctreeOption::ClassNames => {
            FitctreeAction::ClassNames(label_vector(value.clone(), "ClassNames")?.labels)
        }
        FitctreeOption::PredictorNames => {
            FitctreeAction::PredictorNames(text_list(value, "PredictorNames")?)
        }
        FitctreeOption::ResponseName => FitctreeAction::ResponseName(
            scalar_text(value)?
                .ok_or_else(|| fitctree_invalid("fitctree: ResponseName must be text"))?,
        ),
        FitctreeOption::MaxNumSplits => {
            FitctreeAction::MaxNumSplits(nonnegative_integer(value, "MaxNumSplits")?)
        }
        FitctreeOption::MinLeafSize => {
            FitctreeAction::MinLeafSize(positive_integer(value, "MinLeafSize")?)
        }
        FitctreeOption::MinParentSize => {
            FitctreeAction::MinParentSize(positive_integer(value, "MinParentSize")?)
        }
        FitctreeOption::SplitCriterion => {
            let text = scalar_text(value)?
                .ok_or_else(|| fitctree_invalid("fitctree: SplitCriterion must be text"))?;
            let criterion = match text.to_ascii_lowercase().as_str() {
                "gdi" => SplitCriterion::Gdi,
                "deviance" => SplitCriterion::Deviance,
                other => {
                    return Err(fitctree_invalid(format!(
                        "fitctree: unsupported SplitCriterion '{other}'"
                    )));
                }
            };
            FitctreeAction::SplitCriterion(criterion)
        }
        FitctreeOption::Weights => {
            FitctreeAction::Weights(if let Some(name) = scalar_text(value)? {
                WeightSpec::Variable(name)
            } else {
                WeightSpec::Values(numeric_vector(value, "Weights")?)
            })
        }
        FitctreeOption::ScoreTransform => {
            let text = scalar_text(value)?
                .ok_or_else(|| fitctree_invalid("fitctree: ScoreTransform must be text"))?;
            if !text.eq_ignore_ascii_case("none") {
                return Err(fitctree_invalid(
                    "fitctree: only ScoreTransform='none' is supported",
                ));
            }
            FitctreeAction::ScoreTransformNone
        }
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
        | FitctreeOption::Surrogate => {
            return Err(fitctree_invalid(format!(
                "fitctree: option '{}' is not supported yet",
                name.spelling
            )));
        }
    })
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
            None => decode_option(&named.name, &values[named.name.index - index_offset + 1])?,
        };
        match action {
            FitctreeAction::ClassNames(labels) => options.class_names = Some(labels),
            FitctreeAction::PredictorNames(names) => options.predictor_names = Some(names),
            FitctreeAction::ResponseName(name) => options.response_name = Some(name),
            FitctreeAction::MaxNumSplits(count) => options.max_num_splits = Some(count),
            FitctreeAction::MinLeafSize(count) => options.min_leaf_size = count,
            FitctreeAction::MinParentSize(count) => options.min_parent_size = count,
            FitctreeAction::SplitCriterion(criterion) => options.split_criterion = criterion,
            FitctreeAction::Weights(weights) => options.weights = Some(weights),
            FitctreeAction::ScoreTransformNone => {}
        }
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
