use super::super::support::{
    admitted_product_controls, has_literal_zero_axis, inferred_product_shape,
    LiteralReductionOutput, SumOutputClass,
};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, infer_reduction, CallContract, CallInference, CallRequest, DynamicReason,
    InferenceDiagnostic, NumericClass, NumericDomain, NumericFact, ResidencyFact, StorageFact,
    ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let output = match request.arguments.first() {
        None => {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.PROD.INVALID_ARGUMENT",
                "prod requires numerical input data",
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

fn infer_output(
    input: &ValueFact,
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ValueFact {
    if request
        .literals
        .literal_args
        .iter()
        .skip(1)
        .take(request.arguments.len().saturating_sub(1))
        .any(has_literal_zero_axis)
    {
        diagnostics.extend(infer_reduction(input, Some(0)).diagnostics);
        return ValueFact::unknown(DynamicReason::ConflictingControlFlow);
    }
    if input.storage == StorageFact::Sparse {
        diagnostics.push(InferenceDiagnostic::error(
            "RM.PROD.INVALID_INPUT",
            "sparse products are not supported",
        ));
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }
    let (input_class, domain) = match input.kind {
        ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Complex => {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.PROD.COMPLEX_UNSUPPORTED",
                "complex products are not supported",
            ));
            return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(numeric) => (numeric.class, numeric.domain),
        ValueKindFact::Logical => (NumericClass::Double, NumericDomain::Real),
        ValueKindFact::Unknown => return ValueFact::unknown(DynamicReason::RuntimeValue),
        _ => {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.PROD.INVALID_INPUT",
                "expected dense real numeric or logical data",
            ));
            return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    };
    let Some(controls) = admitted_product_controls(request.arguments.len(), &request.literals)
    else {
        if request.literals.literal_args.len() >= request.arguments.len()
            && request
                .literals
                .literal_args
                .iter()
                .skip(1)
                .all(|literal| !matches!(literal, runmat_types::LiteralValue::Unknown))
        {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.PROD.INVALID_ARGUMENT",
                "invalid dimension, missing-data, or output-class controls",
            ));
        }
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    // Runtime empty dimension vectors are accepted only with vector-shaped
    // metadata (1-by-0 or 0-by-1), not the ordinary 0-by-0 empty array.
    for control in request.arguments.iter().skip(1) {
        if control.shape.element_count() == Some(0)
            && control
                .shape
                .known_dims()
                .is_some_and(|dims| dims.len() == 2 && dims.iter().all(|extent| *extent != Some(1)))
        {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.PROD.INVALID_ARGUMENT",
                "empty selector must be vector-shaped",
            ));
            return ValueFact::unknown(DynamicReason::ConflictingControlFlow);
        }
    }
    let shape = inferred_product_shape(&input.shape, &controls);
    let output_policy = match controls.output {
        Some(LiteralReductionOutput::Native) => SumOutputClass::Native,
        Some(LiteralReductionOutput::Double) => SumOutputClass::Double,
        None | Some(LiteralReductionOutput::Default) => SumOutputClass::Default,
    };
    let class = output_policy.resolve_numeric_class(input_class);
    let storage = if input.residency == ResidencyFact::Host {
        match shape.element_count() {
            Some(1) if class == NumericClass::Double && domain == NumericDomain::Real => {
                StorageFact::Scalar
            }
            Some(1) => StorageFact::Unknown,
            Some(_) => StorageFact::Dense,
            None => StorageFact::Unknown,
        }
    } else {
        StorageFact::Unknown
    };
    let mut result = ValueFact::proven(
        if input.kind == ValueKindFact::Logical
            && matches!(controls.output, Some(LiteralReductionOutput::Native))
        {
            ValueKindFact::Logical
        } else {
            ValueKindFact::Numeric(NumericFact { class, domain })
        },
        shape,
        storage,
    );
    if input.residency != ResidencyFact::Host {
        result.residency = ResidencyFact::Unknown;
    }
    if result.shape == input.shape {
        // An ignored trailing-axis selection may return the same backing;
        // shape equality alone cannot prove that any reduction allocated.
        result.alias = runmat_types::AliasFact::Unknown;
    }
    result
}
