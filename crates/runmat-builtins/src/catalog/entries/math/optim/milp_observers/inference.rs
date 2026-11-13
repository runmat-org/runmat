use super::MilpObserverKind;
use crate::BuiltinCatalogEntry;
use crate::OptimObserverState;
use runmat_types::*;
pub(super) fn infer_save(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(MilpObserverKind::SaveSolutions, request, entry)
}
pub(super) fn infer_plot(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(MilpObserverKind::PlotBounds, request, entry)
}
fn infer(
    kind: MilpObserverKind,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut contract = CallContract::fixed(vec![ValueFact::scalar(ValueKindFact::Logical)]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.len() != 3 {
        result.diagnostics.push(InferenceDiagnostic::error(
            "RM-MILP-OBSERVER-ARITY",
            "expected x,optimValues,state",
        ));
    }
    if request.arguments.get(1).is_some_and(|v| {
        !matches!(v.kind, ValueKindFact::Struct(_) | ValueKindFact::Unknown)
            || v.shape.element_count().is_some_and(|n| n != 1)
    }) {
        result.diagnostics.push(InferenceDiagnostic::error(
            "RM-MILP-OBSERVER-VALUES",
            "optimValues must be a scalar struct",
        ));
    }
    let literal_state = request.literals.literal_string_at(2);
    let state = literal_state.as_deref().and_then(OptimObserverState::parse);
    if request.arguments.get(2).is_some_and(|v| {
        !matches!(
            v.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        )
    }) || (literal_state.is_some() && state.is_none())
    {
        result.diagnostics.push(InferenceDiagnostic::error(
            "RM-MILP-OBSERVER-STATE",
            "state must be init, iter, or done",
        ));
    }
    let records_point = kind == MilpObserverKind::SaveSolutions
        && state == Some(OptimObserverState::Iter)
        && request
            .arguments
            .first()
            .is_some_and(|x| x.shape.element_count().is_some_and(|n| n > 0));
    if records_point
        && request.arguments.first().is_some_and(|x| {
            !floating(x)
                || x.shape.known_dims().is_some_and(|dims| {
                    dims.len() > 2 || dims.get(1).is_some_and(|n| n.is_some_and(|n| n != 1))
                })
        })
    {
        result.diagnostics.push(InferenceDiagnostic::error(
            "RM-MILP-OBSERVER-POINT",
            "event x must be a real floating decision column",
        ));
    }
    if let Some(ValueFact {
        kind: ValueKindFact::Struct(values),
        ..
    }) = request.arguments.get(1)
    {
        let fields: &[&str] = match kind {
            MilpObserverKind::PlotBounds => &["time", "fval", "dualbound"],
            MilpObserverKind::SaveSolutions if records_point => &["fval"],
            _ => &[],
        };
        for name in fields {
            match values.fields.get(*name) {
                Some(value)
                    if !matches!(value.kind, ValueKindFact::Unknown)
                        && (!floating(value)
                            || value.shape.element_count().is_some_and(|n| n != 1)) =>
                {
                    result.diagnostics.push(InferenceDiagnostic::error(
                        "RM-MILP-OBSERVER-FIELD",
                        "needed optimValues fields must be floating scalars",
                    ));
                }
                None if values.fields_complete => {
                    result.diagnostics.push(InferenceDiagnostic::error(
                        "RM-MILP-OBSERVER-FIELD",
                        "a needed optimValues field is absent",
                    ))
                }
                _ => {}
            }
        }
    }
    result
}
fn floating(value: &ValueFact) -> bool {
    (matches!(value.kind, ValueKindFact::Unknown)
        || matches!(
            value.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single | NumericClass::Double,
                domain: NumericDomain::Real
            })
        ))
        && value.storage != StorageFact::Sparse
}
