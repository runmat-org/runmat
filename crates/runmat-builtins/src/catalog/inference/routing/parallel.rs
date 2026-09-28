use crate::{BuiltinCatalogEntry, ParallelInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    rule: ParallelInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        ParallelInferenceRule::GetCurrentJob
        | ParallelInferenceRule::GetCurrentTask
        | ParallelInferenceRule::GetCurrentWorker => super::super::unavailable_rule(entry, request),
        _ => super::super::parallel_semantics::infer_parallel_data(request, entry),
    }
}
