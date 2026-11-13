use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    super::super::coordinate_inference::infer(
        request,
        entry,
        1,
        super::super::coordinate_inference::Handles::Scalar,
    )
}
