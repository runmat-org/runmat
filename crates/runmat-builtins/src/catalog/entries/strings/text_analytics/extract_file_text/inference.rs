use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    super::super::extraction_inference::infer(
        request,
        entry,
        super::super::extraction_inference::ExtractionOperation::File,
    )
}
