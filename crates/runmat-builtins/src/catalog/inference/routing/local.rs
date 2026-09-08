use crate::{BuiltinCatalogEntry, BuiltinInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(entry: &BuiltinCatalogEntry, request: &CallRequest) -> CallInference {
    match entry.contract.inference_rule {
        BuiltinInferenceRule::Array(rule) => super::array::infer(rule, request, entry),
        BuiltinInferenceRule::Cells(rule) => super::cells::infer(rule, request, entry),
        BuiltinInferenceRule::Math(rule) => super::math::infer(rule, request, entry),
        BuiltinInferenceRule::Stats(rule) => super::stats::infer(rule, request, entry),
        BuiltinInferenceRule::Acceleration(rule) => {
            super::acceleration::infer(rule, request, entry)
        }
        BuiltinInferenceRule::Aggregate(rule) => super::aggregate::infer(rule, request, entry),
        BuiltinInferenceRule::Introspection(rule) => {
            super::introspection::infer(rule, request, entry)
        }
        BuiltinInferenceRule::Io(rule) => super::io::infer(rule, request, entry),
        BuiltinInferenceRule::Logical(rule) => super::super::logical::infer(rule, request, entry),
        BuiltinInferenceRule::Parallel(rule) => super::parallel::infer(rule, request, entry),
    }
}
