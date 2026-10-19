use super::{CHOL_ERROR_INVALID_ARGUMENT, CHOL_ERROR_INVALID_INPUT};
use crate::catalog::entries::math::linalg::factor::inference::{
    double_scalar, factor_output, matrix_input,
};
use crate::catalog::inference::finish_fixed_outputs;
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, DynamicReason, ValueFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = matrix_input(
        request,
        2,
        true,
        CHOL_ERROR_INVALID_ARGUMENT.code,
        CHOL_ERROR_INVALID_INPUT.code,
    );
    let matrix = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    let factor = factor_output(&matrix, matrix.shape.clone(), false, false, false);
    finish_fixed_outputs(entry, request, vec![factor, double_scalar()], diagnostics)
}
