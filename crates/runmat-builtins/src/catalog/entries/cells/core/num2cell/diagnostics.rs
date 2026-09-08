use crate::catalog::inference::argument_error;
use runmat_types::{CallRequest, InferenceDiagnostic, ValueKindFact};

use super::dimensions;

pub(super) fn for_request(request: &CallRequest) -> Vec<InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-NUM2CELL-ARITY",
            "num2cell expects A and an optional dimension scalar or vector",
            request.arguments.len().saturating_sub(1),
        ));
        return diagnostics;
    }
    if let Some(dim) = request.arguments.get(1) {
        let numeric = matches!(dim.kind, ValueKindFact::Numeric(_) | ValueKindFact::Unknown);
        if !numeric || !dimensions::shape_may_be_vector(&dim.shape) {
            diagnostics.push(argument_error(
                "RM-CATALOG-NUM2CELL-DIM",
                "num2cell dimensions must be a positive integer scalar or vector",
                1,
            ));
        }
        if invalid_literal(request) {
            diagnostics.push(argument_error(
                "RM-CATALOG-NUM2CELL-DIM-VALUE",
                "num2cell dimensions must be unique positive integers within the input rank",
                1,
            ));
        }
    }
    diagnostics
}

fn invalid_literal(request: &CallRequest) -> bool {
    let Some(literal) = request.literals.literal_args.get(1) else {
        return false;
    };
    if dimensions::literal_values(literal).is_err() {
        return true;
    }
    let Some(dims) = dimensions::grouped(request) else {
        return false;
    };
    let mut sorted = dims.clone();
    sorted.sort_unstable();
    if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
        return true;
    }
    request
        .arguments
        .first()
        .and_then(|input| input.shape.rank())
        .is_some_and(|rank| dims.iter().any(|dim| *dim > rank))
}
