mod facts;
mod inference;
mod isfile;
mod isfolder;
mod policy;
mod validation;

use crate::{BuiltinCatalogEntry, PathPredicateInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub use isfile::*;
pub use isfolder::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&ISFILE_CATALOG_ENTRY, &ISFOLDER_CATALOG_ENTRY];

pub(super) fn infer(
    rule: PathPredicateInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    inference::infer(rule, request, entry)
}

fn infer_path_predicate(
    policy: policy::PathPredicateInferencePolicy,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    use crate::catalog::inference::{argument_error, finish_fixed};

    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            policy.arity_code,
            policy.arity_message,
            request.arguments.len().min(1),
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|input| !validation::valid_paths(input))
    {
        diagnostics.push(argument_error(policy.path_code, policy.path_message, 0));
    }
    finish_fixed(
        entry,
        request,
        facts::logical_for_paths(request.arguments.first()),
        diagnostics,
    )
}

#[cfg(test)]
mod tests;
