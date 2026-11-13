use std::collections::BTreeMap;

use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let status = ValueFact::scalar(ValueKindFact::Logical);
    let message = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        BTreeMap::from([
            (
                "message".to_string(),
                ValueFact::scalar(ValueKindFact::String),
            ),
            (
                "identifier".to_string(),
                ValueFact::scalar(ValueKindFact::String),
            ),
        ]),
        true,
    )));
    let mut contract = CallContract::fixed(vec![status, message]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    let argc = request.arguments.len();
    if !(2..=5).contains(&argc) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-XLSWRITE-ARITY",
            "xlswrite requires filename, data, and at most sheet, range, and mode controls",
            0,
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|value| !super::super::super::repl_fs::text_input::scalar(value))
        || request
            .literals
            .literal_string_at(0)
            .is_some_and(|text| text.trim().is_empty())
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-XLSWRITE-PATH",
            "filename must be nonempty scalar text",
            0,
        ));
    }
    if let Some(data) = request.arguments.get(1) {
        let wrong_kind = matches!(
            data.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Complex,
                ..
            }) | ValueKindFact::Struct(_)
                | ValueKindFact::Callable(_)
        );
        let wrong_shape = data.shape.known_dims().is_some_and(|dims| {
            dims.iter()
                .skip(2)
                .any(|extent| extent.is_some_and(|n| n != 1))
        });
        if wrong_kind || wrong_shape {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-XLSWRITE-DATA",
                "data must be a supported two-dimensional spreadsheet value",
                1,
            ));
        }
    }
    result
}
