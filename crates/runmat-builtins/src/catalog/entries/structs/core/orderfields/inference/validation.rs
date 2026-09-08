use crate::catalog::inference::argument_error;
use runmat_types::{CallRequest, InferenceDiagnostic, NumericDomain, ValueFact, ValueKindFact};

pub(super) fn diagnostics(request: &CallRequest) -> Vec<InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    match request.arguments.as_slice() {
        [] => diagnostics.push(argument_error(
            "RM-CATALOG-ORDERFIELDS-ARITY",
            "orderfields requires a structure input",
            0,
        )),
        [target, rest @ ..] => {
            if !valid_target(&target.kind) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ORDERFIELDS-TARGET",
                    "orderfields expects a structure or represented structure array",
                    0,
                ));
            }
            if let Some(order) = rest.first() {
                if !valid_order(order) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-ORDERFIELDS-ORDER",
                        "orderfields expects a reference structure, field-name collection, or real numeric permutation",
                        1,
                    ));
                }
            }
            if rest.len() > 1 {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ORDERFIELDS-ARITY",
                    "orderfields accepts at most two inputs",
                    2,
                ));
            }
        }
    }
    diagnostics
}

fn valid_target(kind: &ValueKindFact) -> bool {
    match kind {
        ValueKindFact::Struct(_) | ValueKindFact::Unknown => true,
        ValueKindFact::Cell(cell) => {
            valid_struct(&cell.element.kind)
                && cell
                    .elements
                    .iter()
                    .all(|element| valid_struct(&element.kind))
        }
        _ => false,
    }
}

fn valid_struct(kind: &ValueKindFact) -> bool {
    matches!(kind, ValueKindFact::Struct(_) | ValueKindFact::Unknown)
}

fn valid_order(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Struct(_)
        | ValueKindFact::Character
        | ValueKindFact::String
        | ValueKindFact::Unknown => true,
        ValueKindFact::Numeric(numeric) => numeric.domain == NumericDomain::Real,
        ValueKindFact::Cell(cell) => {
            valid_order_element(&cell.element.kind)
                && cell
                    .elements
                    .iter()
                    .all(|element| valid_order_element(&element.kind))
        }
        _ => false,
    }
}

fn valid_order_element(kind: &ValueKindFact) -> bool {
    matches!(
        kind,
        ValueKindFact::Struct(_)
            | ValueKindFact::Character
            | ValueKindFact::String
            | ValueKindFact::Unknown
    )
}
