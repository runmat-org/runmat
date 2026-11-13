use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    NumericClass, NumericFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let output = request.arguments.first().map_or_else(
        || ValueFact::unknown(DynamicReason::RuntimeValue),
        |input| match input.numeric() {
            Some(numeric) if numeric.domain == runmat_types::NumericDomain::Real => {
                let class = if matches!(numeric.class, NumericClass::Single) {
                    NumericClass::Single
                } else {
                    NumericClass::Double
                };
                let mut output = input.clone();
                output.kind = ValueKindFact::Numeric(NumericFact {
                    class,
                    domain: numeric.domain,
                });
                output.residency = runmat_types::ResidencyFact::Unknown;
                output.alias = runmat_types::AliasFact::Unique;
                output
            }
            _ => ValueFact::unknown(DynamicReason::RuntimeValue),
        },
    );
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    if request.arguments.len() < 2 {
        inferred.diagnostics.push(InferenceDiagnostic::error(
            "RM.MISSING.INVALID_ARGUMENT",
            "movmad requires data and a window length",
        ));
    }
    inferred
}
