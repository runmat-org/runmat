use crate::*;
use runmat_types::{CallInference, CallRequest, DynamicReason, ValueFact, ValueKindFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    use crate::catalog::entries::math::linalg::factor::inference::{
        extent, factor_output, matrix_shape, schur_operand,
    };
    use crate::catalog::inference::{argument_error, finish_fixed_outputs};
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() || request.arguments.len() > 2 {
        diagnostics.push(argument_error(
            "RM.ORDEIG.INVALID_INPUT",
            "ordeig expects one ordinary or two generalized Schur factors",
            0,
        ));
    }
    let input = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    if !request.arguments.is_empty() {
        diagnostics.extend(schur_operand(&input, 0, "RM.ORDEIG.INVALID_INPUT"));
    }
    if let Some(second) = request.arguments.get(1) {
        diagnostics.extend(schur_operand(second, 1, "RM.ORDEIG.INVALID_INPUT"));
        if let (Some((Some(rows), Some(columns))), Some((Some(other_rows), Some(other_columns)))) = (
            crate::catalog::entries::math::linalg::factor::inference::dimensions(&input),
            crate::catalog::entries::math::linalg::factor::inference::dimensions(second),
        ) {
            if rows != other_rows || columns != other_columns {
                diagnostics.push(argument_error(
                    "RM.ORDEIG.INVALID_INPUT",
                    "generalized Schur factors must have equal order",
                    1,
                ));
            }
        }
    }
    let mut output = factor_output(
        &input,
        matrix_shape(extent(&input), Some(1)),
        false,
        true,
        true,
    );
    // Both ordinary and generalized real forms can have a real or complex
    // spectrum. The result class follows the operands, but the current fact
    // kind cannot encode known class with unknown numerical domain.
    output.kind = ValueKindFact::Unknown;
    finish_fixed_outputs(entry, request, vec![output], diagnostics)
}
