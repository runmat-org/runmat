use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, FactJoin, NumericClass, NumericDomain, NumericFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-UIGETDIR-ARITY",
            "uigetdir accepts an optional starting path and title",
            2,
        ));
    }
    for (index, value) in request.arguments.iter().take(2).enumerate() {
        if !matches!(value.kind, ValueKindFact::Unknown) && !super::super::text_input::scalar(value)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-UIGETDIR-TEXT",
                "uigetdir expects scalar text",
                index,
            ));
        }
    }
    let selected = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    );
    let cancelled = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    finish_fixed_outputs(entry, request, vec![selected.join(&cancelled)], diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::*;
    #[test]
    fn host_result_does_not_falsely_predict_selection_instead_of_cancellation() {
        let request = CallRequest {
            arguments: vec![],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        let result = infer(&request, &super::super::UIGETDIR_CATALOG_ENTRY);
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
        assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
        assert!(result.effects.0.contains(&EffectKind::UserInterface));
        assert!(result.effects.0.contains(&EffectKind::HostCallback));
    }
    #[test]
    fn output_overflow_and_known_invalid_text_are_diagnosed() {
        let request = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::Logical)],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        };
        assert!(
            infer(&request, &super::super::UIGETDIR_CATALOG_ENTRY)
                .diagnostics
                .len()
                >= 2
        );
    }
}
