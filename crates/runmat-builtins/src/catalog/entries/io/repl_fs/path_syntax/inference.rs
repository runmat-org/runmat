use crate::{BuiltinCatalogEntry, PathSyntaxInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    rule: PathSyntaxInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        PathSyntaxInferenceRule::Join => super::fullfile::inference::infer(request, entry),
        PathSyntaxInferenceRule::Split => super::fileparts::inference::infer(request, entry),
        PathSyntaxInferenceRule::FileSeparator => super::filesep::inference::infer(request, entry),
        PathSyntaxInferenceRule::PathListSeparator => {
            super::pathsep::inference::infer(request, entry)
        }
    }
}
