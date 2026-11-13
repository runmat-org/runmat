use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    super::super::row_sort_facts::infer(request, entry, false)
}
