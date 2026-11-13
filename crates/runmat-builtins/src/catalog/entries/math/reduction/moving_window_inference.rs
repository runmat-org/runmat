//! Typed moving-window policies; runtime controls never dispatch by builtin name.
use crate::BuiltinCatalogEntry;
use runmat_types::*;
mod controls;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
pub(super) enum MovingOperation {
    Sum,
    Mean,
    Prod,
    Min,
    Max,
    Median,
    Std,
    Var,
}

impl MovingOperation {
    fn dispersion(self) -> bool {
        matches!(self, Self::Std | Self::Var)
    }
    fn preserve_integer(self) -> bool {
        matches!(self, Self::Min | Self::Max | Self::Median)
    }
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    operation: MovingOperation,
) -> CallInference {
    let output = request.arguments.first().map_or_else(
        || ValueFact::unknown(DynamicReason::RuntimeValue),
        |input| output(input, request, operation),
    );
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    controls::validate_shapes(request, &mut result.diagnostics);
    if request.arguments.len() < 2 {
        result.diagnostics.push(InferenceDiagnostic::error(
            "RM.MOVING_WINDOW.INVALID_ARGUMENT",
            "moving-window reductions require data and a window length",
        ));
    }
    result
}

fn output(input: &ValueFact, request: &CallRequest, operation: MovingOperation) -> ValueFact {
    if input.storage == StorageFact::Sparse {
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }
    let (class, domain) = match input.kind {
        ValueKindFact::Numeric(numeric) => (numeric.class, numeric.domain),
        ValueKindFact::Logical => (NumericClass::Double, NumericDomain::Real),
        _ => return ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    if domain == NumericDomain::Complex && matches!(input.residency, ResidencyFact::Device { .. }) {
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }
    if class.integer_class().is_some()
        && (domain == NumericDomain::Complex || operation.dispersion())
    {
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }
    let class = if class.integer_class().is_some() && !operation.preserve_integer() {
        NumericClass::Double
    } else {
        class
    };
    let domain = if operation.dispersion() {
        NumericDomain::Real
    } else {
        domain
    };
    let shape = controls::shape(&input.shape, request, operation).unwrap_or(ShapeFact::Unknown);
    let storage = if input.residency == ResidencyFact::Host {
        match shape.element_count() {
            Some(1) if class == NumericClass::Double => StorageFact::Scalar,
            Some(1) => StorageFact::Unknown,
            Some(_) => StorageFact::Dense,
            None => StorageFact::Unknown,
        }
    } else {
        StorageFact::Unknown
    };
    let mut output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        shape,
        storage,
    );
    if input.residency != ResidencyFact::Host {
        output.residency = ResidencyFact::Unknown;
    }
    output
}
