mod creation;
mod introspection;

use crate::{ArrayCreationInferenceRule, ArrayInferenceRule, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    rule: ArrayInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        ArrayInferenceRule::Creation(ArrayCreationInferenceRule::Full) => {
            creation::infer_full(request, entry)
        }
        ArrayInferenceRule::Creation(ArrayCreationInferenceRule::Zeros) => {
            creation::infer_zeros(request, entry)
        }
        ArrayInferenceRule::Introspection(rule) => introspection::infer(rule, request, entry),
    }
}
