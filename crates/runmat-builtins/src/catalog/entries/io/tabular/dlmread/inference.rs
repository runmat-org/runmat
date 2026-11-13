use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut matrix = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![None, None]),
        StorageFact::Dense,
    );
    matrix.layout = LayoutFact::ColumnMajor;
    matrix.contiguity = ContiguityFact::Contiguous;
    matrix.view = ViewFact::Materialized;
    matrix.residency = ResidencyFact::Host;
    let mut contract = CallContract::fixed(vec![matrix]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    let len = request.arguments.len();
    if !(1..=5).contains(&len) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-DLMREAD-ARITY",
            "dlmread accepts filename and up to four controls",
            0,
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|value| !super::super::super::repl_fs::text_input::scalar(value))
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-DLMREAD-PATH",
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
            "RM-CATALOG-DLMREAD-PATH",
            "filename must not be empty",
            0,
        ));
    }
    // The tail is overloaded: a scalar number can be a delimiter code or an offset.
    // Reject only geometry that no admitted parser branch can consume.
    for index in 1..len.min(5) {
        let value = &request.arguments[index];
        if matches!(value.kind, ValueKindFact::Character | ValueKindFact::String)
            && !super::super::super::repl_fs::text_input::scalar(value)
        {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-DLMREAD-CONTROL",
                "text controls must be scalar",
                index,
            ));
        }
    }
    result
}
