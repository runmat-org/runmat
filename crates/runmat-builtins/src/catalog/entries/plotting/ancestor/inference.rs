use crate::{
    catalog::inference::{argument_error, finish_fixed, known_invalid_text_scalar},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ANCESTOR-ARITY",
            "expected handle, selector, and optional toplevel",
            0,
        ));
    }
    if let Some(input) = request.arguments.get(1) {
        if !matches!(
            input.kind,
            ValueKindFact::Unknown
                | ValueKindFact::String
                | ValueKindFact::Character
                | ValueKindFact::Cell(_)
        ) || (input.kind == ValueKindFact::Character && known_invalid_text_scalar(input))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ANCESTOR-SELECTOR",
                "selector must contain text type names",
                1,
            ));
        }
    }
    if let Some(option) = request.arguments.get(2) {
        if known_invalid_text_scalar(option)
            || request
                .literals
                .literal_string_at(2)
                .is_some_and(|text| !text.trim().eq_ignore_ascii_case("toplevel"))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ANCESTOR-OPTION",
                "option must be scalar text toplevel",
                2,
            ));
        }
    }
    // Matching depends on the selected registry. A scalar can return a scalar or
    // empty double; a nonscalar input filters unmatched handles into a row.
    let source = request.arguments.first();
    let empty = source.is_some_and(|input| {
        input.shape.element_count() == Some(0)
            || input
                .numeric()
                .is_some_and(|n| n.class != NumericClass::Double || n.domain != NumericDomain::Real)
            || !matches!(
                input.kind,
                ValueKindFact::Unknown | ValueKindFact::Numeric(_)
            )
    });
    let shape = if empty {
        ShapeFact::from(vec![Some(0), Some(0)])
    } else {
        ShapeFact::Ranked { rank: 2 }
    };
    let output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        if empty {
            StorageFact::Dense
        } else {
            StorageFact::Unknown
        },
    );
    finish_fixed(entry, request, output, diagnostics)
}
