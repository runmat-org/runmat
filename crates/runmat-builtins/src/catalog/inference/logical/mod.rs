mod relational;

use crate::{BuiltinCatalogEntry, LogicalInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    rule: LogicalInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        LogicalInferenceRule::NumericClassification(predicate) => {
            super::numeric_classification::infer(request, entry, predicate)
        }
        LogicalInferenceRule::MetadataPredicate(predicate) => {
            super::metadata_predicate::infer(request, entry, predicate)
        }
        LogicalInferenceRule::Relational(operator) => relational::infer(request, entry, operator),
        LogicalInferenceRule::ScalarReduction(reduction) => {
            super::scalar_logical_reduction::infer(request, entry, reduction)
        }
    }
}
