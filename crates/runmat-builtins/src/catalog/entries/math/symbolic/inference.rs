//! Scalar symbolic success facts and exact per-form operand roles.
use super::scalar_contract::SymbolicScalarOperation;
use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
mod operands;
mod literals;
#[cfg(test)]
mod tests;
pub(super) fn infer(
    operation: SymbolicScalarOperation,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let count = request.arguments.len();
    let mut diagnostics = Vec::new();
    if !operation.accepts_arity(count) {
        diagnostics.push(argument_error(
            operation.arity_error().code,
            "inputs do not match the supported scalar symbolic forms",
            count,
        ));
    }
    let unsupported = operation == SymbolicScalarOperation::Integrate
        && literals::unsupported_int_option(request);
    if unsupported {
        diagnostics.push(argument_error(
            entry.descriptor.errors[3].code,
            "this symbolic integration option is recognized but not implemented",
            count - 2,
        ));
    }
    for (index, argument) in request.arguments.iter().enumerate() {
        if unsupported && index >= count - 2 {
            continue;
        }
        let role = operation.operand(index, count);
        if !operands::permits(argument, role)
            || request
                .literals
                .literal_args
                .get(index)
                .is_some_and(|v| !literals::valid(operation, index, count, v))
        {
            diagnostics.push(argument_error(
                operation.operand_error(index, count).code,
                "operand does not match this host scalar symbolic role",
                index,
            ));
        }
    }
    let kind = if operation == SymbolicScalarOperation::Precision {
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    } else {
        ValueKindFact::Symbolic
    };
    let mut output = ValueFact::scalar(kind);
    if operation != SymbolicScalarOperation::Precision {
        output.alias = AliasFact::Unknown;
    }
    let mut result = finish_fixed(entry, request, output, diagnostics);
    if operation == SymbolicScalarOperation::Decimal && count == 2 {
        result.effects.0.remove(&EffectKind::EnvironmentRead);
    }
    if operation == SymbolicScalarOperation::Precision && count == 0 {
        result.effects.0.remove(&EffectKind::EnvironmentWrite);
    }
    result
}
