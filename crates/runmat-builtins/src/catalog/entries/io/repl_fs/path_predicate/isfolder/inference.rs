use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

use super::super::policy::PathPredicateInferencePolicy;

const POLICY: PathPredicateInferencePolicy = PathPredicateInferencePolicy {
    arity_code: "RM-CATALOG-ISFOLDER-ARITY",
    arity_message: "isfolder expects exactly one input",
    path_code: "RM-CATALOG-ISFOLDER-PATH",
    path_message: "isfolder expects a string array, character row, or cell array of character rows",
};

pub(in crate::catalog::entries::io::repl_fs::path_predicate) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    super::super::infer_path_predicate(POLICY, request, entry)
}
