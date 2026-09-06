use crate::{BuiltinCatalogEntry, SearchPathInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(super) mod input;
pub(super) mod result;

pub(super) fn infer(
    rule: SearchPathInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        SearchPathInferenceRule::QueryOrReplace => super::path::inference::infer(request, entry),
        SearchPathInferenceRule::Add => super::addpath::inference::infer(request, entry),
        SearchPathInferenceRule::Remove => super::rmpath::inference::infer(request, entry),
    }
}
