use super::super::support::{admitted_mean_controls, inferred_mean_shape, LiteralReductionOutput};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    NumericClass, NumericDomain, NumericFact, ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let output = match request.arguments.first() {
        None => {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.MEAN.INVALID_ARGUMENT",
                "mean requires numerical input data",
            ));
            ValueFact::unknown(DynamicReason::RuntimeValue)
        }
        Some(input) => infer_output(input, request, &mut diagnostics),
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

pub(super) fn infer_fusion_fact(request: &CallRequest) -> Option<ValueFact> {
    let input = request.arguments.first()?;
    // The public rule renders this rejected carrier as a diagnostic. A
    // finite facts-only request must reject it before that String/Vec exists.
    if matches!(input.kind, ValueKindFact::Numeric(numeric)
        if numeric.domain == NumericDomain::Complex
            && numeric.class.integer_class().is_some())
    {
        return None;
    }
    let mut diagnostics = Vec::new();
    let fact = infer_output(input, request, &mut diagnostics);
    diagnostics.is_empty().then_some(fact)
}

fn infer_output(
    input: &ValueFact,
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ValueFact {
    if input.storage == StorageFact::Sparse {
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }
    let (input_class, domain) = match input.kind {
        ValueKindFact::Numeric(numeric)
            if numeric.domain == NumericDomain::Complex
                && numeric.class.integer_class().is_some() =>
        {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.MEAN.INVALID_INPUT",
                "complex integer input is not supported",
            ));
            return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(numeric) => (numeric.class, numeric.domain),
        ValueKindFact::Logical => (NumericClass::Double, NumericDomain::Real),
        ValueKindFact::Unknown => return ValueFact::unknown(DynamicReason::RuntimeValue),
        _ => {
            // The runtime retains a historical reversed-argument normalization;
            // this catalog does not prove such calls from the leading operand.
            return ValueFact::unknown(DynamicReason::RuntimeValue);
        }
    };
    let Some(controls) = admitted_mean_controls(request.arguments.len(), &request.literals) else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let class = match controls.output {
        Some(LiteralReductionOutput::Native) => input_class,
        Some(LiteralReductionOutput::Double) => NumericClass::Double,
        Some(LiteralReductionOutput::Default) | None if input_class == NumericClass::Single => {
            NumericClass::Single
        }
        Some(LiteralReductionOutput::Default) | None => NumericClass::Double,
    };
    let shape = inferred_mean_shape(&input.shape, &controls);
    let storage = if input.residency != ResidencyFact::Host {
        StorageFact::Unknown
    } else {
        match shape.element_count() {
            Some(1) if class == NumericClass::Double && domain == NumericDomain::Real => {
                StorageFact::Scalar
            }
            Some(1) => StorageFact::Unknown,
            Some(_) => StorageFact::Dense,
            None => StorageFact::Unknown,
        }
    };
    let mut result = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        shape,
        storage,
    );
    if input.residency != ResidencyFact::Host {
        result.residency = ResidencyFact::Unknown;
    }
    result
}
