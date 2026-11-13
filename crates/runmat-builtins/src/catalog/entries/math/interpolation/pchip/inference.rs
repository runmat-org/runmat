use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    super::super::piecewise_inference::infer(
        request,
        entry,
        super::super::piecewise_inference::PiecewiseFamily::Pchip,
    )
}
