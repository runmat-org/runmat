use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    super::super::moving_window_inference::infer(
        request,
        entry,
        super::super::moving_window_inference::MovingOperation::Median,
    )
}
