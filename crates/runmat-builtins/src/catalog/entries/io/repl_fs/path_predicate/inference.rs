use crate::{BuiltinCatalogEntry, PathPredicateInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    rule: PathPredicateInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        PathPredicateInferenceRule::File => super::isfile::inference::infer(request, entry),
        PathPredicateInferenceRule::Folder => super::isfolder::inference::infer(request, entry),
    }
}
