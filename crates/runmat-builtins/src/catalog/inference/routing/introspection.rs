use crate::{BuiltinCatalogEntry, IntrospectionInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    rule: IntrospectionInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        IntrospectionInferenceRule::Feval => {
            super::super::introspection_semantics::infer_feval(request, entry)
        }
    }
}
