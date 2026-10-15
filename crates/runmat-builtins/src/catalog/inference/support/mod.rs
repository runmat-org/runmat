use super::super::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, InferenceDiagnostic, LiteralValue,
    NumericClass, NumericDomain, NumericFact, ResidencyFact, ValueFact, ValueKindFact,
};

pub(super) mod facts;
pub(super) mod materialization;
pub(super) mod object_handle;
pub(super) mod text;

pub(crate) fn literal_text(literal: &LiteralValue) -> Option<String> {
    match literal {
        LiteralValue::String(value)
        | LiteralValue::Character(value)
        | LiteralValue::Keyword(value) => Some(value.clone()),
        _ => None,
    }
}

pub(crate) fn numeric_kind(class: NumericClass, domain: NumericDomain) -> ValueKindFact {
    ValueKindFact::Numeric(NumericFact { class, domain })
}

pub(crate) fn preserved_binary_residency(
    left: &ResidencyFact,
    right: &ResidencyFact,
) -> ResidencyFact {
    match (left, right) {
        (ResidencyFact::Host, ResidencyFact::Host) => ResidencyFact::Host,
        (
            ResidencyFact::Device {
                provider: left_owner,
            },
            ResidencyFact::Device {
                provider: right_owner,
            },
        ) if left_owner == right_owner => left.clone(),
        (ResidencyFact::Device { .. }, ResidencyFact::Host) => left.clone(),
        (ResidencyFact::Host, ResidencyFact::Device { .. }) => right.clone(),
        _ => ResidencyFact::Unknown,
    }
}

pub(crate) fn default_double_scalar() -> ValueFact {
    ValueFact::scalar(numeric_kind(NumericClass::Double, NumericDomain::Real))
}

pub(crate) fn finish_fixed(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
    output: ValueFact,
    diagnostics: Vec<InferenceDiagnostic>,
) -> CallInference {
    finish_fixed_outputs(entry, request, vec![output], diagnostics)
}

pub(crate) fn finish_fixed_outputs(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
    outputs: Vec<ValueFact>,
    mut diagnostics: Vec<InferenceDiagnostic>,
) -> CallInference {
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}

pub(crate) fn argument_error(
    code: impl Into<String>,
    message: impl Into<String>,
    argument: usize,
) -> InferenceDiagnostic {
    let mut diagnostic = InferenceDiagnostic::error(code, message);
    diagnostic.argument = Some(argument);
    diagnostic
}
