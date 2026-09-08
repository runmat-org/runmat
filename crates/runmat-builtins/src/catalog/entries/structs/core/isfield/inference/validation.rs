use crate::catalog::inference::argument_error;
use runmat_types::{CallRequest, DimensionFact, InferenceDiagnostic, ShapeFact, ValueKindFact};

pub(super) fn diagnostics(request: &CallRequest) -> Vec<InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-ISFIELD-ARITY",
            "isfield requires exactly two inputs",
            request.arguments.len().min(1),
        ));
    }
    if let Some(names) = request.arguments.get(1) {
        if !valid_names(&names.kind, &names.shape) {
            diagnostics.push(argument_error(
                "RM-CATALOG-ISFIELD-NAMES",
                "isfield expects a character row, string value, string array, or cell collection of scalar text",
                1,
            ));
        }
    }
    diagnostics
}

fn valid_names(kind: &ValueKindFact, shape: &ShapeFact) -> bool {
    match kind {
        ValueKindFact::Character => row_or_unknown(shape),
        ValueKindFact::String | ValueKindFact::Unknown => true,
        ValueKindFact::Cell(cell) => {
            valid_cell_element(&cell.element.kind)
                && cell
                    .elements
                    .iter()
                    .all(|element| valid_cell_element(&element.kind))
        }
        _ => false,
    }
}

fn valid_cell_element(kind: &ValueKindFact) -> bool {
    matches!(
        kind,
        ValueKindFact::Character | ValueKindFact::String | ValueKindFact::Unknown
    )
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
