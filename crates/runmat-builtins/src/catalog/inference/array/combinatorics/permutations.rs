use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

use super::super::super::{argument_error, finish_fixed, support};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-PERMS-ARITY",
            "perms requires exactly one input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-PERMS-ARITY",
            "perms accepts exactly one input",
            1,
        ));
    }

    let mut output = input.clone();
    match &output.kind {
        ValueKindFact::Numeric(_)
        | ValueKindFact::Logical
        | ValueKindFact::Character
        | ValueKindFact::String
        | ValueKindFact::Cell(_) => {
            output.shape = permutation_shape(&input.shape);
            if let ValueKindFact::Cell(cell) = &mut output.kind {
                cell.elements.clear();
                cell.elements_complete = false;
            }
            support::facts::materialize(&mut output);
        }
        ValueKindFact::Unknown => {
            output.shape = ShapeFact::Ranked { rank: 2 };
            output.storage = StorageFact::Unknown;
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-PERMS-INPUT",
                "perms requires a supported scalar or vector",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn permutation_shape(input: &ShapeFact) -> ShapeFact {
    let Some(elements) = input.element_count() else {
        return ShapeFact::Ranked { rank: 2 };
    };
    let Some(rows) = checked_factorial(elements) else {
        return ShapeFact::Ranked { rank: 2 };
    };
    ShapeFact::from(vec![Some(rows), Some(elements)])
}

fn checked_factorial(value: usize) -> Option<usize> {
    (1..=value).try_fold(1_usize, usize::checked_mul)
}
