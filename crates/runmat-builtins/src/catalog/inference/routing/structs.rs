use crate::{BuiltinCatalogEntry, StructInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    rule: StructInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        StructInferenceRule::Structfun => {
            crate::catalog::entries::structs::core::structfun::infer(request, entry)
        }
    }
}
