/// `fitctree` name-value identities, decoded once at the public boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitctreeOption {
    ClassNames,
    PredictorNames,
    ResponseName,
    MaxNumSplits,
    MinLeafSize,
    MinParentSize,
    SplitCriterion,
    Weights,
    ScoreTransform,
    AlgorithmForCategorical,
    CategoricalPredictors,
    Cost,
    CrossVal,
    CvPartition,
    Holdout,
    KFold,
    Leaveout,
    MaxNumCategories,
    MergeLeaves,
    NumBins,
    NumVariablesToSample,
    OptimizeHyperparameters,
    HyperparameterOptimizationOptions,
    PredictorSelection,
    Prior,
    Prune,
    PruneCriterion,
    Reproducible,
    Surrogate,
}

impl FitctreeOption {
    pub fn from_public_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "classnames" => Self::ClassNames,
            "predictornames" => Self::PredictorNames,
            "responsename" => Self::ResponseName,
            "maxnumsplits" => Self::MaxNumSplits,
            "minleafsize" => Self::MinLeafSize,
            "minparentsize" => Self::MinParentSize,
            "splitcriterion" => Self::SplitCriterion,
            "weights" => Self::Weights,
            "scoretransform" => Self::ScoreTransform,
            "algorithmforcategorical" => Self::AlgorithmForCategorical,
            "categoricalpredictors" => Self::CategoricalPredictors,
            "cost" => Self::Cost,
            "crossval" => Self::CrossVal,
            "cvpartition" => Self::CvPartition,
            "holdout" => Self::Holdout,
            "kfold" => Self::KFold,
            "leaveout" => Self::Leaveout,
            "maxnumcategories" => Self::MaxNumCategories,
            "mergeleaves" => Self::MergeLeaves,
            "numbins" => Self::NumBins,
            "numvariablestosample" => Self::NumVariablesToSample,
            "optimizehyperparameters" => Self::OptimizeHyperparameters,
            "hyperparameteroptimizationoptions" => Self::HyperparameterOptimizationOptions,
            "predictorselection" => Self::PredictorSelection,
            "prior" => Self::Prior,
            "prune" => Self::Prune,
            "prunecriterion" => Self::PruneCriterion,
            "reproducible" => Self::Reproducible,
            "surrogate" => Self::Surrogate,
            _ => return None,
        })
    }

    pub const fn supported(self) -> bool {
        matches!(
            self,
            Self::ClassNames
                | Self::PredictorNames
                | Self::ResponseName
                | Self::MaxNumSplits
                | Self::MinLeafSize
                | Self::MinParentSize
                | Self::SplitCriterion
                | Self::Weights
                | Self::ScoreTransform
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn option_names_have_closed_typed_identity() {
        assert_eq!(
            FitctreeOption::from_public_name("MINLEAFSIZE"),
            Some(FitctreeOption::MinLeafSize)
        );
        assert_eq!(
            FitctreeOption::from_public_name("Prune"),
            Some(FitctreeOption::Prune)
        );
        assert!(!FitctreeOption::Prune.supported());
        assert!(FitctreeOption::Weights.supported());
        assert_eq!(FitctreeOption::from_public_name("notAnOption"), None);
    }
}
