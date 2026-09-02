use crate::{ArrayInferenceRule, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    rule: ArrayInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        ArrayInferenceRule::Full => super::super::array_semantics::infer_full(request, entry),
        ArrayInferenceRule::ShapePredicate(predicate) => {
            super::super::shape_predicate::infer(request, entry, predicate)
        }
        ArrayInferenceRule::ShapeScalarQuery(query) => {
            super::super::shape_scalar_query::infer(request, entry, query)
        }
        ArrayInferenceRule::Zeros => super::super::array_semantics::infer_zeros(request, entry),
    }
}
