use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, NumericClass,
    NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

use super::super::scalar_facts;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FZERO-ARITY",
            "fzero requires a function and scalar point or two-element bracket",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 4)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-FZERO-OUTPUTS",
            "fzero returns at most four outputs",
            4,
        ));
    }
    scalar_facts::callback(request, &mut diagnostics);
    scalar_facts::real_numeric(request.arguments.get(1), 1, &[1, 2], &mut diagnostics);
    scalar_facts::options(request.arguments.get(2), 2, &mut diagnostics);
    let class = match request.arguments.get(1).map(|value| &value.kind) {
        Some(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })) => Some(NumericClass::Single),
        Some(
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
            | ValueKindFact::Logical,
        ) => Some(NumericClass::Double),
        _ => None,
    };
    let numeric_result = || match class {
        Some(class) => scalar_facts::host_scalar(class),
        None => {
            let mut fact = scalar_facts::conditional_host_scalar();
            fact.shape = ShapeFact::Scalar;
            fact.storage = StorageFact::Scalar;
            fact.residency = ResidencyFact::Host;
            fact
        }
    };
    let outputs = match request.outputs.requested.known_count() {
        Some(count) => (0..count)
            .map(|index| match index {
                0..=2 => numeric_result(),
                3 => scalar_facts::diagnostic_struct(),
                _ => ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            })
            .collect(),
        None => vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
    };
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    if let Some(callback) = scalar_facts::callback_invocation(request, numeric_result()) {
        contract.effects.0.extend(callback.effects.0);
        contract.capabilities.0.extend(callback.capabilities.0);
        diagnostics.extend(callback.diagnostics);
    }
    let mut inferred = infer_call(&contract, request);
    inferred.dynamic_outputs = request.outputs.requested.known_count().is_none();
    if !diagnostics.is_empty() {
        inferred.outputs.iter_mut().for_each(|output| {
            *output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
        });
    }
    inferred.diagnostics.extend(diagnostics);
    inferred
}
