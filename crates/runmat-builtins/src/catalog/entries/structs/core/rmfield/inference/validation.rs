use crate::catalog::inference::argument_error;
use runmat_types::{
    CallRequest, DimensionFact, InferenceDiagnostic, ShapeFact, ValueFact, ValueKindFact,
};

pub(super) fn diagnostics(request: &CallRequest) -> Vec<InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-RMFIELD-ARITY",
            "rmfield requires a structure and at least one field-name input",
            request.arguments.len().min(1),
        ));
    }
    if let Some(target) = request.arguments.first() {
        if !valid_target(&target.kind) {
            diagnostics.push(argument_error(
                "RM-CATALOG-RMFIELD-TARGET",
                "rmfield expects a structure or represented structure array",
                0,
            ));
        }
    }
    for (index, names) in request.arguments.iter().enumerate().skip(1) {
        if !valid_names(names) {
            diagnostics.push(argument_error(
                "RM-CATALOG-RMFIELD-NAMES",
                "rmfield expects character rows, string values, string arrays, or cell collections of scalar text",
                index,
            ));
        }
    }
    diagnostics
}

fn valid_target(kind: &ValueKindFact) -> bool {
    match kind {
        ValueKindFact::Struct(_) | ValueKindFact::Unknown => true,
        ValueKindFact::Cell(cell) => {
            valid_struct_element(&cell.element.kind)
                && cell
                    .elements
                    .iter()
                    .all(|element| valid_struct_element(&element.kind))
        }
        _ => false,
    }
}

fn valid_struct_element(kind: &ValueKindFact) -> bool {
    matches!(kind, ValueKindFact::Struct(_) | ValueKindFact::Unknown)
}

fn valid_names(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Character => row_or_unknown(&value.shape),
        ValueKindFact::String | ValueKindFact::Unknown => true,
        ValueKindFact::Cell(cell) => {
            valid_scalar_text(&cell.element) && cell.elements.iter().all(valid_scalar_text)
        }
        _ => false,
    }
}

fn valid_scalar_text(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::Character => row_or_unknown(&value.shape),
        ValueKindFact::String | ValueKindFact::Unknown => {
            value.is_scalar() || value.shape == ShapeFact::Unknown
        }
        _ => false,
    }
}

fn row_or_unknown(shape: &ShapeFact) -> bool {
    match shape {
        ShapeFact::Scalar | ShapeFact::Unknown | ShapeFact::Ranked { .. } => true,
        ShapeFact::Shaped { dims } => {
            matches!(
                dims.first(),
                None | Some(DimensionFact::Known(1) | DimensionFact::Unknown)
            )
        }
    }
}
