use crate::{BuiltinCatalogEntry, IndexingInferenceRule};
use runmat_types::{CallInference, CallRequest};

mod common;
mod find;
mod ind2sub;
mod sub2ind;
#[cfg(test)]
mod tests;

pub(super) fn infer(
    rule: IndexingInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        IndexingInferenceRule::Find => find::infer_find(request, entry),
        IndexingInferenceRule::Ind2Sub => ind2sub::infer_ind2sub(request, entry),
        IndexingInferenceRule::Sub2Ind => sub2ind::infer_sub2ind(request, entry),
    }
}
