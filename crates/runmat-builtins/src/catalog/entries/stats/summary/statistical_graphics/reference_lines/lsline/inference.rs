//! Sound least-squares-line handle facts and axes-form admission.
use super::super::facts;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, InferenceDiagnostic, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 1 {
        diagnostics.push(InferenceDiagnostic::error(
            "ARITY",
            "lsline accepts only an optional axes handle",
        ));
    }
    if let Some(axes) = request.arguments.first() {
        if axes.shape.element_count().is_some_and(|count| count != 1)
            || !matches!(
                axes.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Unknown
            )
        {
            diagnostics.push(InferenceDiagnostic::error(
                "AXES",
                "lsline requires a scalar axes handle",
            ));
        }
    }
    // Every successful route returns host numeric line handles. A scalar
    // numeric fact cannot prove a live axes identity or the handle count.
    let proven = diagnostics.is_empty();
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
