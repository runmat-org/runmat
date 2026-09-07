use crate::catalog::inference::argument_error;
use runmat_types::{
    broadcast_shape, CallRequest, InferenceDiagnostic, NumericDomain, ShapeFact, ValueFact,
    ValueKindFact,
};

pub(super) fn arguments(request: &CallRequest) -> Vec<InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-BSXFUN-ARITY",
            "bsxfun requires a binary callable and two array inputs",
            request.arguments.len().min(2),
        ));
    }
    if let Some(function) = request.arguments.first() {
        if !matches!(
            function.kind,
            ValueKindFact::Callable(_) | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-BSXFUN-FUNCTION",
                "bsxfun requires a callable first argument",
                0,
            ));
        }
    }
    for (index, input) in request.arguments.iter().enumerate().skip(1).take(2) {
        if !supported(input) {
            diagnostics.push(argument_error(
                "RM-CATALOG-BSXFUN-INPUT",
                "bsxfun inputs must be numeric, logical, complex, or character arrays",
                index,
            ));
        }
    }
    diagnostics
}

pub(super) fn output_shape(
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ShapeFact {
    let (Some(left), Some(right)) = (request.arguments.get(1), request.arguments.get(2)) else {
        return ShapeFact::Unknown;
    };
    match broadcast_shape(&left.shape, &right.shape) {
        Ok(shape) => shape,
        Err(_) => {
            diagnostics.push(argument_error(
                "RM-CATALOG-BSXFUN-SIZE",
                "bsxfun inputs are not compatible for singleton expansion",
                2,
            ));
            ShapeFact::Unknown
        }
    }
}

fn supported(input: &ValueFact) -> bool {
    match input.kind {
        ValueKindFact::Numeric(numeric) => {
            numeric.domain == NumericDomain::Real || numeric.class.integer_class().is_none()
        }
        ValueKindFact::Logical | ValueKindFact::Character | ValueKindFact::Unknown => true,
        _ => false,
    }
}
