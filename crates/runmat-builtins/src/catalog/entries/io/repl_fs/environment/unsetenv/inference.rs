use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

use super::super::{facts, validation};

pub(in crate::catalog::entries::io::repl_fs::environment) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-UNSETENV-ARITY",
            "unsetenv expects exactly one input",
            request.arguments.len().min(1),
        ));
    }
    if let Some(input) = request.arguments.first() {
        if !validation::valid_name(input, validation::NamePolicy::Matlab) {
            diagnostics.push(argument_error(
                "RM-CATALOG-UNSETENV-NAME",
                "unsetenv expects a text name or text container",
                0,
            ));
        }
    }
    finish_fixed(entry, request, facts::double_scalar(), diagnostics)
}
