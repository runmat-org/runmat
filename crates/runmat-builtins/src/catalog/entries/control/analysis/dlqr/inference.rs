use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    use super::super::care::inference::{infer_standard, StandardRiccatiApi};
    infer_standard(request, entry, StandardRiccatiApi::Dlqr)
}
