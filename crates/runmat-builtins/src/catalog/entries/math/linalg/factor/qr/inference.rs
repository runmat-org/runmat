use super::{QR_ERROR_INVALID_ARGUMENT, QR_ERROR_INVALID_INPUT};
use crate::catalog::entries::math::linalg::factor::inference::{
    binary64_output, dimensions, literal_factor_option, matrix_input, matrix_shape,
};
use crate::catalog::inference::finish_fixed_outputs;
use crate::BuiltinCatalogEntry;
use crate::FactorOption;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericDomain, ShapeFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = matrix_input(
        request,
        3,
        false,
        QR_ERROR_INVALID_ARGUMENT.code,
        QR_ERROR_INVALID_INPUT.code,
    );
    let matrix = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    let (rows, columns) = dimensions(&matrix).unwrap_or((None, None));
    let options_are_absent = request.arguments.len() == 1;
    let q_shape = if options_are_absent {
        matrix_shape(rows, rows)
    } else {
        ShapeFact::Ranked { rank: 2 }
    };
    let r_shape = if options_are_absent {
        matrix_shape(rows, columns)
    } else {
        ShapeFact::Ranked { rank: 2 }
    };
    let mut pivot = Some(FactorOption::Matrix);
    for index in 1..request.arguments.len() {
        match literal_factor_option(request, index) {
            Some(FactorOption::Vector) => pivot = Some(FactorOption::Vector),
            Some(FactorOption::Matrix) => pivot = Some(FactorOption::Matrix),
            Some(FactorOption::Economy) => {}
            Some(_) | None => {
                pivot = None;
                break;
            }
        }
    }
    let permutation_shape = match pivot {
        Some(FactorOption::Vector) => matrix_shape(columns, Some(1)),
        Some(FactorOption::Matrix) => matrix_shape(columns, columns),
        _ => ShapeFact::Ranked { rank: 2 },
    };
    let mut q = binary64_output(&matrix, q_shape, false, false);
    let mut r = binary64_output(&matrix, r_shape, false, false);
    if matches!(&matrix.kind, ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Complex)
    {
        // The host QR value converter may erase numerically zero imaginary
        // parts, while a provider may retain complex storage.
        q.kind = ValueKindFact::Unknown;
        r.kind = ValueKindFact::Unknown;
    }
    let permutation = binary64_output(&matrix, permutation_shape, true, false);
    let first = match request.outputs.requested.known_count() {
        Some(1) => r.clone(),
        Some(_) => q,
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed_outputs(entry, request, vec![first, r, permutation], diagnostics)
}
