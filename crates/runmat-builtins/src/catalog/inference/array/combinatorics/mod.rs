mod cartesian_product;
mod coefficient_class;
mod permutations;
mod selection_combinations;

#[cfg(test)]
mod tests;

use crate::{BuiltinCatalogEntry, CombinatoricsInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    rule: CombinatoricsInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        CombinatoricsInferenceRule::CartesianProduct => cartesian_product::infer(request, entry),
        CombinatoricsInferenceRule::Permutations => permutations::infer(request, entry),
        CombinatoricsInferenceRule::SelectionCombinations => {
            selection_combinations::infer(request, entry)
        }
    }
}
