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
    if request.arguments.len() < 2 {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-DLMWRITE-ARITY",
            "dlmwrite requires filename and matrix",
            0,
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|value| !super::super::super::repl_fs::text_input::scalar(value))
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-DLMWRITE-PATH",
            "filename must be scalar text",
            0,
        ));
    }
    if request
        .literals
        .literal_string_at(0)
        .is_some_and(|text| text.trim().is_empty())
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-DLMWRITE-PATH",
            "filename must not be empty",
            0,
        ));
    }
    if let Some(data) = request.arguments.get(1) {
        let wrong_kind = !matches!(
            data.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) | ValueKindFact::Logical
                | ValueKindFact::Unknown
        );
        let wrong_shape = data.shape.known_dims().is_some_and(|dims| {
            dims.iter()
                .skip(2)
                .any(|extent| extent.is_some_and(|n| n != 1))
        });
        if wrong_kind || wrong_shape {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-DLMWRITE-DATA",
                "data must be a real numeric or logical matrix",
                1,
            ));
        }
    }
    result
}
