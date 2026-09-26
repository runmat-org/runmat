use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{infer_call, CallContract, CallInference, CallRequest};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-RUN-ARITY",
            "run requires one script path",
            0,
        ));
    }
    if let Some(path) = request.arguments.first() {
        if !matches!(path.kind, runmat_types::ValueKindFact::Unknown)
            && !super::super::text_input::scalar(path)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-RUN-SCRIPT",
                "run expects a character row or string scalar",
                0,
            ));
        }
    }
    let mut contract = CallContract::fixed(Vec::new());
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    diagnostics.append(&mut result.diagnostics);
    result.diagnostics = diagnostics;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::*;

    #[test]
    fn unknown_source_preserves_dynamic_effects_without_invented_outputs() {
        let request = CallRequest {
            arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(0)),
        };
        let result = infer(&request, &super::super::RUN_CATALOG_ENTRY);
        assert!(result.diagnostics.is_empty());
        assert!(result.outputs.is_empty());
        assert!(result.effects.0.contains(&EffectKind::Unknown));
        assert!(result.effects.0.contains(&EffectKind::WorkspaceWrite));
    }

    #[test]
    fn rejects_known_invalid_paths_and_requested_outputs() {
        let request = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::Logical)],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        let result = infer(&request, &super::super::RUN_CATALOG_ENTRY);
        assert!(result.diagnostics.len() >= 2);
    }

    #[test]
    fn unknown_kind_with_row_shape_can_still_be_text() {
        let request = CallRequest {
            arguments: vec![ValueFact::proven(
                ValueKindFact::Unknown,
                ShapeFact::from(vec![Some(1), Some(5)]),
                StorageFact::Unknown,
            )],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(0)),
        };
        assert!(infer(&request, &super::super::RUN_CATALOG_ENTRY)
            .diagnostics
            .is_empty());
    }

    #[test]
    fn canonical_semantics_preserve_dynamic_workspace_execution() {
        let entry = crate::builtin_catalog_entry_by_name("run").unwrap();
        let semantics = entry.legacy_semantics();
        assert_eq!(
            semantics.workspace_effect,
            Some(crate::BuiltinWorkspaceEffect::LoadsExternalBindings)
        );
        assert!(semantics.effects.workspace);
        assert!(semantics.effects.filesystem);
        assert!(semantics.effects.unknown);
        assert!(crate::builtin_semantics_for_name("run").is_none());
    }
}
