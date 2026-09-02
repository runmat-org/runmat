use crate::{AggregateInferenceRule, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    rule: AggregateInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        AggregateInferenceRule::Struct => {
            super::super::aggregate_semantics::infer_struct_builtin(request, entry)
        }
    }
}
