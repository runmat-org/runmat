use super::super::inference_support::{environment, infer_with};
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-PYTHON-ARITY",
            "terminate requires one PythonEnvironment",
            0,
        ));
    }
    if let Some(argument) = request.arguments.first() {
        let possible = match &argument.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => object
                .runtime_class
                .as_ref()
                .is_none_or(|class| class.is(super::super::PYTHON_ENVIRONMENT_CLASS)),
            _ => false,
        };
        if !possible {
            diagnostics.push(argument_error(
                "RM-PYTHON-ENVIRONMENT",
                "terminate expects a PythonEnvironment returned by pyenv",
                0,
            ));
        }
    }
    infer_with(
        request,
        entry,
        CallContract::fixed(vec![environment()]),
        diagnostics,
    )
}
