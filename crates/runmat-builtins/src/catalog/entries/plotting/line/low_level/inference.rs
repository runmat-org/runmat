use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    super::super::super::coordinate_inference::infer(
        request,
        entry,
        0,
        super::super::super::coordinate_inference::Handles::Series,
    )
}
