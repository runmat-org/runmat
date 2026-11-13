use super::super::reorder_facts::{self, Reorder};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    reorder_facts::infer(request, entry, Reorder::Flip)
}
