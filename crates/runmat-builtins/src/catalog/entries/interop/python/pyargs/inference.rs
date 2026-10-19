use super::super::inference_support::{bundle, infer_with, pairs};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    pairs(request, 0, &mut diagnostics, "pyargs");
    infer_with(
        request,
        entry,
        CallContract::fixed(vec![bundle(request)]),
        diagnostics,
    )
}
