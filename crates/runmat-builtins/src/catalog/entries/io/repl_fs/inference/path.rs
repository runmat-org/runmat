use crate::{BuiltinCatalogEntry, PathInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    rule: PathInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        PathInferenceRule::Installation(rule) => {
            super::super::installation_path::infer(rule, request, entry)
        }
        PathInferenceRule::Predicate(rule) => {
            super::super::path_predicate::infer(rule, request, entry)
        }
        PathInferenceRule::Syntax(rule) => super::super::path_syntax::infer(rule, request, entry),
        PathInferenceRule::Search(rule) => super::super::search_path::infer(rule, request, entry),
        PathInferenceRule::Temporary(rule) => {
            super::super::temporary_path::infer(rule, request, entry)
        }
    }
}
