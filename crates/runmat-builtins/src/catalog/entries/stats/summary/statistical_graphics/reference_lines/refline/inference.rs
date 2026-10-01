//! Sound reference-line handle facts for admitted coefficient forms.
use super::super::facts;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, InferenceDiagnostic, NumericDomain,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if let Some(first) = request.arguments.first() {
        if !matches!(
            first.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
        ) || first
            .numeric()
            .is_some_and(|number| number.domain != NumericDomain::Real)
        {
            diagnostics.push(InferenceDiagnostic::error(
                "COEFFICIENT",
                "refline requires real numeric coefficients or an axes handle",
            ));
        }
        if first
            .shape
            .element_count()
            .is_some_and(|count| count != 1 && count != 2)
        {
            diagnostics.push(InferenceDiagnostic::error(
                "SHAPE",
                "refline coefficient input must be scalar or have two elements",
            ));
        }
    }
    // Two scalar numbers remain ambiguous: the first might be a live axes
    // handle. A single two-element numeric vector is not an axes handle.
    let coefficient_vector = request.arguments.len() == 1
        && request.arguments.first().is_some_and(|fact| {
            matches!(fact.kind, ValueKindFact::Numeric(_))
                && fact
                    .numeric()
                    .is_some_and(|n| n.domain == NumericDomain::Real)
                && fact.shape.element_count() == Some(2)
        });
    let proven = diagnostics.is_empty() && (request.arguments.is_empty() || coefficient_vector);
    let mut contract = CallContract::fixed(vec![if proven {
        facts::handles()
    } else {
        facts::dynamic()
    }]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
