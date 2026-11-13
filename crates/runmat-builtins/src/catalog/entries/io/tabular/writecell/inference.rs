use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let bytes = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut contract = CallContract::fixed(vec![bytes]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    let argc = request.arguments.len();
    if argc < 2 || !(argc - 2).is_multiple_of(2) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-WRITECELL-ARITY",
            "writecell requires a cell array, filename, and complete name-value pairs",
            0,
        ));
    }
    if let Some(data) = request.arguments.first() {
        let wrong_kind = !matches!(data.kind, ValueKindFact::Cell(_) | ValueKindFact::Unknown);
        let wrong_shape = data.shape.known_dims().is_some_and(|dims| {
            dims.iter()
                .skip(2)
                .any(|extent| extent.is_some_and(|n| n != 1))
        });
        if wrong_kind || wrong_shape {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-WRITECELL-DATA",
                "input must be a two-dimensional cell array",
                0,
            ));
        }
    }
    if request
        .arguments
        .get(1)
        .is_some_and(|value| !super::super::super::repl_fs::text_input::scalar(value))
        || request
            .literals
            .literal_string_at(1)
            .is_some_and(|text| text.trim().is_empty())
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-WRITECELL-PATH",
            "filename must be nonempty scalar text",
            1,
        ));
    }
    for index in (2..argc).step_by(2) {
        if !super::super::super::repl_fs::text_input::scalar(&request.arguments[index]) {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-WRITECELL-OPTION",
                "option name must be scalar text",
                index,
            ));
        }
    }
    result
}
