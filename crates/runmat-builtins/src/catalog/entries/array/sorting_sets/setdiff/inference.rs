use super::super::set_facts::{binary_options, finish_binary, ElementShape, ResultClass};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let rows = binary_options(request, &mut diagnostics);
    finish_binary(
        request,
        entry,
        rows,
        ElementShape::PreserveVector,
        ResultClass::Left,
        2,
        diagnostics,
    )
}
