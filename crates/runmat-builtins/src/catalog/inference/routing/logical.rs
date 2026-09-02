use crate::{BuiltinCatalogEntry, LogicalInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    rule: LogicalInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        LogicalInferenceRule::NumericClassification(predicate) => {
            super::super::numeric_classification::infer(request, entry, predicate)
        }
        LogicalInferenceRule::MetadataPredicate(predicate) => {
            super::super::metadata_predicate::infer(request, entry, predicate)
        }
        LogicalInferenceRule::ScalarReduction(reduction) => {
            super::super::scalar_logical_reduction::infer(request, entry, reduction)
        }
    }
}
