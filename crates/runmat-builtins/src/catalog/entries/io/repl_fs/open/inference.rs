use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::{CallInference, CallRequest, DynamicReason, ValueFact, ValueKindFact};
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-OPEN-ARITY",
            "open expects one target name",
            0,
        ));
    }
    if let Some(name) = request.arguments.first() {
        if !matches!(name.kind, ValueKindFact::Unknown) && !super::super::text_input::scalar(name) {
            diagnostics.push(argument_error(
                "RM-CATALOG-OPEN-NAME",
                "open expects a character row or string scalar",
                0,
            ));
        }
    }
    if request
        .literals
        .literal_string_at(0)
        .is_some_and(|name| name.is_empty())
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-OPEN-EMPTY",
            "open target name must not be empty",
            0,
        ));
    }
    // Even a literal filename can resolve to a workspace variable or a user
    // extension handler. Its output is not statically constrained to host data.
    finish_fixed_outputs(
        entry,
        request,
        vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        diagnostics,
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::*;
    #[test]
    fn zero_outputs_preserve_callback_effects_and_unknown_output_is_not_host_asserted() {
        for count in [0, 1] {
            let request = CallRequest {
                arguments: vec![ValueFact::scalar(ValueKindFact::String)],
                literals: LiteralContext::new(vec![LiteralValue::String("custom.abc".into())]),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
            };
            let result = infer(&request, &super::super::OPEN_CATALOG_ENTRY);
            assert!(result.diagnostics.is_empty());
            assert!(result.effects.0.contains(&EffectKind::HostCallback));
            assert!(result.effects.0.contains(&EffectKind::Unknown));
            if count == 1 {
                assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
            }
        }
    }
    #[test]
    fn rejects_proven_invalid_text_and_excess_outputs() {
        let request = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::Logical)],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        };
        assert!(
            infer(&request, &super::super::OPEN_CATALOG_ENTRY)
                .diagnostics
                .len()
                >= 2
        );
    }
}
