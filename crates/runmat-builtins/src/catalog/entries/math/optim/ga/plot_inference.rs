use super::super::nonlinear_facts as facts;
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;
pub(super) fn infer(r: &CallRequest, e: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if r.arguments.len() != 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-GA-PLOT-ARITY",
            "expected options,state,flag",
            0,
        ));
    }
    facts::options(r.arguments.first(), 0, true, &mut diagnostics);
    if let Some(state) = r.arguments.get(1) {
        if !matches!(
            state.kind,
            ValueKindFact::Unknown | ValueKindFact::Struct(_)
        ) || state.shape.element_count().is_some_and(|n| n != 1)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-GA-PLOT-STATE",
                "genetic plot state must be a scalar structure",
                1,
            ));
        }
    }
    if r.literals
        .literal_string_at(2)
        .is_some_and(|flag| crate::OptimGeneticObserverState::parse(&flag).is_none())
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-GA-PLOT-FLAG",
            "flag must be init,iter,interrupt,done",
            2,
        ));
    }
    let state = r
        .arguments
        .get(1)
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    facts::finish(r, e, vec![state], diagnostics, None)
}
