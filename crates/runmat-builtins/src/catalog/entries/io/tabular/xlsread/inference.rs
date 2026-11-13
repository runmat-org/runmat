use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut numeric = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![None, None]),
        StorageFact::Dense,
    );
    numeric.layout = LayoutFact::ColumnMajor;
    numeric.contiguity = ContiguityFact::Contiguous;
    numeric.view = ViewFact::Materialized;
    numeric.residency = ResidencyFact::Host;
    let cell = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(vec![None, None]),
        StorageFact::Dense,
    );
    let mut contract = CallContract::fixed(vec![numeric, cell.clone(), cell]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    let argc = request.arguments.len();
    if !(1..=4).contains(&argc) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-XLSREAD-ARITY",
            "xlsread accepts filename and at most three selectors",
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
            "RM-CATALOG-XLSREAD-FILENAME",
            "filename must be nonempty scalar text",
            0,
        ));
    }
    if argc == 4
        && request.literals.literal_string_at(3).is_some_and(|mode| {
            !matches!(
                mode.trim().to_ascii_lowercase().as_str(),
                "basic" | "-basic"
            )
        })
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-XLSREAD-MODE",
            "fourth input must be '-basic'",
            3,
        ));
    }
    result
}
