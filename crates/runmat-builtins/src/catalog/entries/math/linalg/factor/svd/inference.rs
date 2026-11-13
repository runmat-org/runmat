use super::{SVD_ERROR_INVALID_ARGUMENT, SVD_ERROR_INVALID_INPUT};
use crate::catalog::entries::math::linalg::factor::inference::{
    binary64_output, dimensions, literal_factor_option, matrix_input, matrix_shape, minimum_extent,
};
use crate::catalog::inference::finish_fixed_outputs;
use crate::BuiltinCatalogEntry;
use crate::FactorOption;
use runmat_types::{CallInference, CallRequest, DynamicReason, ShapeFact, StorageFact, ValueFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = matrix_input(
        request,
        3,
        false,
        SVD_ERROR_INVALID_ARGUMENT.code,
        SVD_ERROR_INVALID_INPUT.code,
    );
    let matrix = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    let (rows, columns) = dimensions(&matrix).unwrap_or((None, None));
    let inner = minimum_extent(&matrix);
    let values_shape = if inner == Some(0) {
        matrix_shape(Some(0), Some(0))
    } else if inner == Some(1) {
        ShapeFact::Scalar
    } else {
        matrix_shape(inner, Some(1))
    };
    let mut values = binary64_output(&matrix, values_shape, true, true);
    if inner == Some(1) {
        // singular_vector_value uses tensor_into_value for one element.
        values.storage = StorageFact::Scalar;
    }
    let no_options = request.arguments.len() == 1;
    let u = binary64_output(
        &matrix,
        if no_options {
            matrix_shape(rows, rows)
        } else {
            ShapeFact::Ranked { rank: 2 }
        },
        false,
        true,
    );
    let singular_matrix = binary64_output(
        &matrix,
        if no_options {
            matrix_shape(rows, columns)
        } else {
            ShapeFact::Ranked { rank: 2 }
        },
        true,
        true,
    );
    let v = binary64_output(
        &matrix,
        if no_options {
            matrix_shape(columns, columns)
        } else {
            ShapeFact::Ranked { rank: 2 }
        },
        false,
        true,
    );
    let mut sigma_format = Some(FactorOption::Matrix);
    for index in 1..request.arguments.len() {
        match literal_factor_option(request, index) {
            Some(FactorOption::Vector) => sigma_format = Some(FactorOption::Vector),
            Some(FactorOption::Matrix) => sigma_format = Some(FactorOption::Matrix),
            Some(FactorOption::Economy | FactorOption::Full) => {}
            Some(_) | None => {
                sigma_format = None;
                break;
            }
        }
    }
    let second = match sigma_format {
        Some(FactorOption::Vector) => values.clone(),
        Some(FactorOption::Matrix) => singular_matrix,
        _ => binary64_output(&matrix, ShapeFact::Ranked { rank: 2 }, true, true),
    };
    let first = match request.outputs.requested.known_count() {
        Some(1) => values,
        Some(_) => u,
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed_outputs(entry, request, vec![first, second, v], diagnostics)
}
