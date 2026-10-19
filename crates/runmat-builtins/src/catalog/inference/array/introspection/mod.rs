mod shape_predicate;
mod shape_query;
mod shape_scalar_query;

use crate::{ArrayIntrospectionInferenceRule, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    rule: ArrayIntrospectionInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        ArrayIntrospectionInferenceRule::ShapePredicate(predicate) => {
            shape_predicate::infer(request, entry, predicate)
        }
        ArrayIntrospectionInferenceRule::ShapeQuery(query) => {
            shape_query::infer(request, entry, query)
        }
        ArrayIntrospectionInferenceRule::ShapeScalarQuery(query) => {
            shape_scalar_query::infer(request, entry, query)
        }
    }
}
