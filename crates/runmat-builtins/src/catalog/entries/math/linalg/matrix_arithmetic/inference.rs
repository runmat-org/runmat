use crate::catalog::inference::{argument_error, finish_fixed, materialize};
use crate::{BuiltinCatalogEntry, MatrixArithmeticInferenceRule};
use runmat_types::{
    infer_binary, CallInference, CallRequest, DynamicReason, OperatorKind, ValueFact,
};

pub(super) fn infer(
    rule: MatrixArithmeticInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-MATRIX-ARITHMETIC-ARITY",
            "matrix arithmetic requires exactly two inputs",
            request.arguments.len().min(2),
        ));
    }
    let (Some(left), Some(right)) = (request.arguments.first(), request.arguments.get(1)) else {
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    let operator = match rule {
        MatrixArithmeticInferenceRule::Multiply => OperatorKind::MatrixMultiply,
        MatrixArithmeticInferenceRule::LeftDivide => OperatorKind::Mldivide,
        MatrixArithmeticInferenceRule::RightDivide => OperatorKind::Mrdivide,
        MatrixArithmeticInferenceRule::Power => OperatorKind::MatrixPower,
    };
    let inferred = infer_binary(operator, left, right);
    diagnostics.extend(inferred.diagnostics);
    let mut output = inferred.fact;
    materialize(&mut output);
    finish_fixed(entry, request, output, diagnostics)
}
