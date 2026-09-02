use crate::{BuiltinCatalogEntry, ParallelInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    rule: ParallelInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        ParallelInferenceRule::Parpool => {
            super::super::parallel_semantics::infer_parallel_pool(request, entry, false)
        }
        ParallelInferenceRule::Gcp => {
            super::super::parallel_semantics::infer_parallel_pool(request, entry, true)
        }
        ParallelInferenceRule::Parfeval | ParallelInferenceRule::ParfevalOnAll => {
            super::super::parallel_semantics::infer_parallel_future(request, entry)
        }
        ParallelInferenceRule::FetchOutputs => {
            super::super::parallel_semantics::infer_parallel_fetch(request, entry, false)
        }
        ParallelInferenceRule::FetchNext => {
            super::super::parallel_semantics::infer_parallel_fetch(request, entry, true)
        }
        ParallelInferenceRule::GetCurrentJob
        | ParallelInferenceRule::GetCurrentTask
        | ParallelInferenceRule::GetCurrentWorker => super::super::unavailable_rule(entry, request),
        _ => super::super::parallel_semantics::infer_parallel_data(request, entry),
    }
}
