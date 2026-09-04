mod index_labels;

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
        GroupingInferenceRule::IndexLabels => index_labels::infer(request, entry),
    }
}
