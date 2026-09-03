mod real_unary;

use crate::{BuiltinCatalogEntry, ErrorFunctionInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    rule: ErrorFunctionInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    real_unary::infer(rule, request, entry)
}
