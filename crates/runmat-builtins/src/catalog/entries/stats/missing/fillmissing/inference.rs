use super::FillMissingMethodKind;
use crate::catalog::inference::literal_text;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, FactJoin,
    InferenceDiagnostic, NumericClass, NumericDomain, NumericFact, ResidencyFact, StorageFact,
    ValueFact, ValueKindFact,
};

fn result_fact(request: &CallRequest, input: &ValueFact) -> ValueFact {
    if unsupported_array_representation(input) {
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }
    match &input.kind {
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::String => {
            let mut output = input.clone();
            output.residency = ResidencyFact::Host;
            output.alias = runmat_types::AliasFact::Unique;
            // A missing scalar carrier returns the constant itself; a tensor
            // converts it into its own class. Facts may describe either carrier.
            let scalar_can_be_missing =
                matches!(
                    input.kind,
                    ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Double,
                        domain: NumericDomain::Real | NumericDomain::Complex,
                    }) | ValueKindFact::String
                ) && matches!(input.storage, StorageFact::Scalar | StorageFact::Unknown)
                    && input.shape.element_count().is_none_or(|count| count == 1);
            let constant_possible = request
                .literals
                .literal_args
                .get(1)
                .and_then(literal_text)
                .is_none_or(|method| {
                    FillMissingMethodKind::parse(&method) == Some(FillMissingMethodKind::Constant)
                });
            if scalar_can_be_missing && constant_possible {
                if let Some(fill) = request.arguments.get(2) {
                    // The scalar route returns the supplied value directly;
                    // unlike the input, that value is not gathered here.
                    return output.join(fill);
                }
            }
            output
        }
        // Tables can contain variables with different classes and masks.
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    }
}

fn unsupported_array_representation(input: &ValueFact) -> bool {
    input.storage == StorageFact::Sparse
        || input
            .numeric()
            .is_some_and(|numeric| numeric.domain == NumericDomain::Complex)
            && input.storage != StorageFact::Scalar
}

fn mask_fact(input: &ValueFact) -> ValueFact {
    if unsupported_array_representation(input) {
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }
    match &input.kind {
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::String => {
            // The runtime returns a LogicalArray even for scalar input; a
            // singleton shape does not turn that dense carrier into Bool.
            ValueFact::proven(
                ValueKindFact::Logical,
                input.shape.clone(),
                StorageFact::Dense,
            )
        }
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let (result, mask) = request.arguments.first().map_or_else(
        || {
            (
                ValueFact::unknown(DynamicReason::RuntimeValue),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            )
        },
        |input| (result_fact(request, input), mask_fact(input)),
    );
    let mut contract = CallContract::fixed(vec![result, mask]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    if request.arguments.len() < 2 {
        inferred.diagnostics.push(InferenceDiagnostic::error(
            "RM.MISSING.INVALID_ARGUMENT",
            "fillmissing requires data and a fill method",
        ));
    }
    inferred
}
