use super::super::inference_support::{environment, infer_with, pairs, text_at};
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    pairs(request, 0, &mut diagnostics, "pyenv");
    for index in (0..request.arguments.len()).step_by(2) {
        if let Some(name) = request.literals.literal_string_at(index) {
            if !name.eq_ignore_ascii_case("Version") && !name.eq_ignore_ascii_case("ExecutionMode")
            {
                diagnostics.push(argument_error(
                    "RM-PYTHON-OPTION",
                    "pyenv supports Version and ExecutionMode",
                    index,
                ));
            }
        }
        text_at(request, index + 1, &mut diagnostics, "pyenv option value");
    }
    let mut result = infer_with(
        request,
        entry,
        CallContract::fixed(vec![environment()]),
        diagnostics,
    );
    if request.arguments.is_empty() {
        result.effects.0.remove(&EffectKind::EnvironmentWrite);
    }
    result
}
