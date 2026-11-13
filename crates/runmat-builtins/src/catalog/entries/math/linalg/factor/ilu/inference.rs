use super::{ILU_ERROR_INVALID_ARGUMENT, ILU_ERROR_INVALID_INPUT};
use crate::catalog::entries::math::linalg::factor::inference::dimensions;
use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, NumericFact,
    ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let count = request.outputs.requested.known_count();
    if request.arguments.is_empty() || request.arguments.len() > 2 || count.is_some_and(|v| v > 3) {
        diagnostics.push(argument_error(
            ILU_ERROR_INVALID_ARGUMENT.code,
            ILU_ERROR_INVALID_ARGUMENT.message,
            request.arguments.len().min(2),
        ));
    }
    let input = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    if !matches!(input.storage, StorageFact::Sparse | StorageFact::Unknown) {
        diagnostics.push(argument_error(
            ILU_ERROR_INVALID_INPUT.code,
            ILU_ERROR_INVALID_INPUT.message,
            0,
        ));
    }
    if input.shape.rank().is_some_and(|rank| rank != 2)
        || dimensions(&input)
            .is_some_and(|(rows, columns)| rows.zip(columns).is_some_and(|(a, b)| a != b))
    {
        diagnostics.push(argument_error(
            ILU_ERROR_INVALID_INPUT.code,
            ILU_ERROR_INVALID_INPUT.message,
            0,
        ));
    }
    let kind = match input.kind {
        ValueKindFact::Numeric(f)
            if matches!(f.class, NumericClass::Single | NumericClass::Double) =>
        {
            ValueKindFact::Numeric(f)
        }
        ValueKindFact::Unknown => ValueKindFact::Unknown,
        _ => {
            diagnostics.push(argument_error(
                ILU_ERROR_INVALID_INPUT.code,
                ILU_ERROR_INVALID_INPUT.message,
                0,
            ));
            ValueKindFact::Unknown
        }
    };
    let output = ValueFact {
        kind,
        shape: match input.shape {
            ShapeFact::Scalar => ShapeFact::from(vec![Some(1), Some(1)]),
            s => s,
        },
        storage: StorageFact::Sparse,
        residency: ResidencyFact::Host,
        ..ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    finish_fixed_outputs(
        entry,
        request,
        vec![
            output.clone(),
            output.clone(),
            ValueFact {
                kind: ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                ..output
            },
        ],
        diagnostics,
    )
}
