use crate::{ArrayInferenceRule, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    rule: ArrayInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    super::super::array::infer(rule, request, entry)
}
