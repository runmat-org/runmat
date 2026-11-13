use crate::*;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    super::super::model_inference::infer(
        request,
        entry,
        super::super::model_inference::ModelOperation::Words,
    )
}
