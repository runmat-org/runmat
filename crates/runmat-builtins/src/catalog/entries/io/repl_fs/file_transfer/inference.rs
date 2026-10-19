use crate::{BuiltinCatalogEntry, FileTransferInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    rule: FileTransferInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        FileTransferInferenceRule::Copy => super::copyfile::inference::infer(request, entry),
        FileTransferInferenceRule::Move => super::movefile::inference::infer(request, entry),
    }
}
