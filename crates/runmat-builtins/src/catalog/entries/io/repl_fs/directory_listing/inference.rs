use crate::{BuiltinCatalogEntry, DirectoryListingInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    rule: DirectoryListingInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        DirectoryListingInferenceRule::Metadata => super::dir::inference::infer(request, entry),
        DirectoryListingInferenceRule::Names => super::ls::inference::infer(request, entry),
    }
}
