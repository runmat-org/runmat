use crate::{BuiltinCatalogEntry, SourceInventoryInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    rule: SourceInventoryInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        SourceInventoryInferenceRule::FolderContents => {
            super::what::inference::infer(request, entry)
        }
    }
}
