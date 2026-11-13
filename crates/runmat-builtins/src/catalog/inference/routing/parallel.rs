use crate::{BuiltinCatalogEntry, ParallelInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    _rule: ParallelInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    super::super::parallel_semantics::infer_parallel_data(request, entry)
}
