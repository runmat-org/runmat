mod counts;
mod grouped_apply;
mod index_labels;
mod roles;
mod sorted_groups;

#[cfg(test)]
mod tests;

use crate::{BuiltinCatalogEntry, GroupingInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    rule: GroupingInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        GroupingInferenceRule::Counts => counts::infer(request, entry),
        GroupingInferenceRule::GroupedApply => grouped_apply::infer(request, entry),
        GroupingInferenceRule::IndexLabels => index_labels::infer(request, entry),
        GroupingInferenceRule::SortedGroups => sorted_groups::infer(request, entry),
    }
}
