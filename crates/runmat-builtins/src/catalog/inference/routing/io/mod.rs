use crate::{BuiltinCatalogEntry, IoInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    rule: IoInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    crate::catalog::entries::io::infer(rule, request, entry)
}
