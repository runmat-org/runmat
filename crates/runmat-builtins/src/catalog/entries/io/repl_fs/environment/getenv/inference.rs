use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, DynamicReason, ValueFact, ValueKindFact};

use super::super::{facts, validation};

pub(in crate::catalog::entries::io::repl_fs::environment) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-GETENV-ARITY",
            "getenv accepts at most one input",
            1,
        ));
    }
    let output = match request.arguments.first() {
        None => facts::dictionary(),
        Some(input) => {
            if !validation::valid_name(input, validation::NamePolicy::GetenvExtensions) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-GETENV-NAME",
                    "getenv expects a text name or text container",
                    0,
                ));
                ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
            } else if matches!(input.kind, ValueKindFact::Unknown) {
                ValueFact::unknown(DynamicReason::DynamicDispatch)
            } else {
                facts::text_for_names(input)
            }
        }
    };
    finish_fixed(entry, request, output, diagnostics)
}
