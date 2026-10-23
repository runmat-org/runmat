use super::{ICHOL_ERROR_INVALID_ARGUMENT, ICHOL_ERROR_INVALID_INPUT};
use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, ResidencyFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty()
        || request.arguments.len() > 2
        || request
            .outputs
            .requested
            .known_count()
            .is_some_and(|count| count > 1)
    {
        diagnostics.push(argument_error(
            ICHOL_ERROR_INVALID_ARGUMENT.code,
            ICHOL_ERROR_INVALID_ARGUMENT.message,
            0,
        ));
    }
    let input = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    if input.storage != StorageFact::Sparse && input.storage != StorageFact::Unknown {
        diagnostics.push(argument_error(
            ICHOL_ERROR_INVALID_INPUT.code,
            ICHOL_ERROR_INVALID_INPUT.message,
            0,
        ));
    }
    if let Some(dims) = input.shape.known_dims() {
        if dims.len() != 2 || matches!((dims[0], dims[1]), (Some(rows), Some(cols)) if rows != cols)
        {
            diagnostics.push(argument_error(
                ICHOL_ERROR_INVALID_INPUT.code,
                ICHOL_ERROR_INVALID_INPUT.message,
                0,
            ));
        }
    }
    let class = match &input.kind {
        ValueKindFact::Numeric(fact)
            if matches!(fact.class, NumericClass::Single | NumericClass::Double) =>
        {
            ValueKindFact::Numeric(*fact)
        }
        ValueKindFact::Unknown => ValueKindFact::Unknown,
        _ => {
            diagnostics.push(argument_error(
                ICHOL_ERROR_INVALID_INPUT.code,
                ICHOL_ERROR_INVALID_INPUT.message,
                0,
            ));
            ValueKindFact::Unknown
        }
    };
    let output = ValueFact {
        kind: class,
        shape: match input.shape {
            ShapeFact::Scalar => ShapeFact::from(vec![Some(1), Some(1)]),
            other => other,
        },
        storage: StorageFact::Sparse,
        residency: ResidencyFact::Host,
        ..ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    finish_fixed_outputs(entry, request, vec![output], diagnostics)
}
