mod accumulation;
mod binning;
mod combinatorics;
mod creation;
mod grouping;
mod introspection;

use crate::{ArrayCreationInferenceRule, ArrayInferenceRule, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    rule: ArrayInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        ArrayInferenceRule::Accumulation(rule) => accumulation::infer(rule, request, entry),
        ArrayInferenceRule::Binning(rule) => binning::infer(rule, request, entry),
        ArrayInferenceRule::Combinatorics(rule) => combinatorics::infer(rule, request, entry),
        ArrayInferenceRule::Creation(ArrayCreationInferenceRule::Full) => {
            creation::infer_full(request, entry)
        }
        ArrayInferenceRule::Creation(ArrayCreationInferenceRule::Zeros) => {
            creation::infer_zeros(request, entry)
        }
        ArrayInferenceRule::Grouping(rule) => grouping::infer(rule, request, entry),
        ArrayInferenceRule::Introspection(rule) => introspection::infer(rule, request, entry),
    }
}
