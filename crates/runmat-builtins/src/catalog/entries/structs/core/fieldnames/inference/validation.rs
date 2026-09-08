use crate::catalog::inference::argument_error;
use runmat_types::{CallRequest, InferenceDiagnostic, ValueKindFact};

pub(super) fn diagnostics(request: &CallRequest) -> Vec<InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    match request.arguments.as_slice() {
        [] => diagnostics.push(argument_error(
            "RM-CATALOG-FIELDNAMES-ARITY",
            "fieldnames requires one input",
            0,
        )),
        [input, rest @ ..] => {
            if !valid_target(&input.kind) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FIELDNAMES-TARGET",
                    "fieldnames expects a structure, represented structure array, or supported object",
                    0,
                ));
            }
            if !rest.is_empty() {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FIELDNAMES-ARITY",
                    "fieldnames accepts exactly one input",
                    1,
                ));
            }
        }
    }
    diagnostics
}

fn valid_target(kind: &ValueKindFact) -> bool {
    match kind {
        ValueKindFact::Struct(_) | ValueKindFact::Object(_) | ValueKindFact::Unknown => true,
        ValueKindFact::Cell(cell) => {
            matches!(
                cell.element.kind,
                ValueKindFact::Struct(_) | ValueKindFact::Unknown
            ) && cell.elements.iter().all(|element| {
                matches!(
                    element.kind,
                    ValueKindFact::Struct(_) | ValueKindFact::Unknown
                )
            })
        }
        _ => false,
    }
}
