use crate::{AccelerationInferenceRule, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    rule: AccelerationInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        AccelerationInferenceRule::Gather => {
            super::super::acceleration_semantics::infer_gather(request, entry)
        }
        AccelerationInferenceRule::GpuArray => {
            super::super::acceleration_semantics::infer_gpu_array(request, entry)
        }
    }
}
