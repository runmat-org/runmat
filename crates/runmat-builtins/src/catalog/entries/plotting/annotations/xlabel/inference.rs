use super::super::shared::{self, AnnotationForm};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    shared::infer(request, entry, AnnotationForm::AxesNumericText)
}
