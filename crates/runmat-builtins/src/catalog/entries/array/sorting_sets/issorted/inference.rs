use super::super::sort_facts::{self, SortForm};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    sort_facts::infer(request, entry, SortForm::Predicate)
}
