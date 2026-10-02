use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, NumericClass,
    NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

use super::super::scalar_facts;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(3..=4).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FMINBND-ARITY",
            "fminbnd requires a function and two bounds, with optional options",
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
            "RM-CATALOG-FMINBND-OUTPUTS",
            "fminbnd returns at most four outputs",
            4,
        ));
    }
    scalar_facts::callback(request, &mut diagnostics);
    scalar_facts::real_numeric(request.arguments.get(1), 1, &[1], &mut diagnostics);
    scalar_facts::real_numeric(request.arguments.get(2), 2, &[1], &mut diagnostics);
    scalar_facts::options(request.arguments.get(3), 3, &mut diagnostics);

    // x1 > x2 is an admitted call with empty x and fval, not a scalar
    // result. Unknown bound values therefore do not prove scalar geometry.
    let known_consistent = request
        .literals
        .numeric_at(1)
        .zip(request.literals.numeric_at(2))
        .is_some_and(|(lower, upper)| lower.is_finite() && upper.is_finite() && lower <= upper);
    let numeric_result = || {
        let mut result = if known_consistent {
            scalar_facts::host_scalar(NumericClass::Double)
        } else {
            ValueFact::unknown(DynamicReason::RuntimeValue)
        };
        result.kind = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
        result.residency = ResidencyFact::Host;
        if !known_consistent {
            result.shape = ShapeFact::Unknown;
            result.storage = StorageFact::Unknown;
        }
        result
    };
    let outputs = match request.outputs.requested.known_count() {
        Some(count) => (0..count)
            .map(|index| match index {
                0 | 1 => numeric_result(),
                2 => scalar_facts::host_scalar(NumericClass::Double),
                3 => scalar_facts::diagnostic_struct(),
                _ => ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            })
            .collect(),
        None => vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
    };
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    if let Some(callback) =
        scalar_facts::callback_invocation(request, scalar_facts::host_scalar(NumericClass::Double))
    {
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
