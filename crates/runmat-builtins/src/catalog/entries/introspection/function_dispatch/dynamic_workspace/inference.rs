use crate::{
    catalog::inference::{argument_error, known_invalid_text_scalar},
    BuiltinCatalogEntry,
};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, ValueFact, ValueKindFact,
};

pub(super) fn infer_eval(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_source_call(request, entry, 1, &[0], false)
}

pub(super) fn infer_evalc(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_source_call(request, entry, 1, &[0], true)
}

pub(super) fn infer_evalin(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_source_call(request, entry, 2, &[0, 1], false)
}

pub(super) fn infer_assignin(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-ASSIGNIN-ARITY",
            "assignin requires a workspace, variable name, and value",
            0,
        ));
    }
    for index in [0, 1] {
        if request
            .arguments
            .get(index)
            .is_some_and(invalid_scalar_text)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ASSIGNIN-TEXT",
                "workspace selector and variable name must be scalar text",
                index,
            ));
        }
    }
    let mut contract = CallContract::fixed(Vec::new());
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}

fn infer_source_call(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    arity: usize,
    text_positions: &[usize],
    captured_text: bool,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != arity {
        diagnostics.push(argument_error(
            "RM-CATALOG-DYNAMIC-WORKSPACE-ARITY",
            "source evaluation requires the declared scalar-text arguments",
            0,
        ));
    }
    for &index in text_positions {
        if request
            .arguments
            .get(index)
            .is_some_and(invalid_scalar_text)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-DYNAMIC-WORKSPACE-TEXT",
                "source and workspace selector must be scalar text",
                index,
            ));
        }
    }
    // Evaluated source is opaque at this boundary: a requested result never
    // proves a class, shape, resident owner, or output count. evalc alone has
    // a known first-output kind, but its captured text length remains dynamic.
    let mut contract = CallContract::dynamic(DynamicReason::RuntimeValue);
    if captured_text {
        let mut text = ValueFact::unknown(DynamicReason::RuntimeValue);
        text.kind = ValueKindFact::String;
        contract.outputs.push(text);
    }
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}

fn invalid_scalar_text(value: &ValueFact) -> bool {
    !matches!(
        &value.kind,
        ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
    ) || known_invalid_text_scalar(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{EffectKind, LiteralContext, OutputSelection, RequestedOutputCount};

    fn request(arguments: Vec<ValueFact>, outputs: RequestedOutputCount) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(outputs),
        }
    }

    #[test]
    fn source_evaluation_retains_workspace_effects_without_inventing_results() {
        let source = ValueFact::unknown(DynamicReason::RuntimeValue);
        let eval = infer_eval(
            &request(vec![source.clone()], RequestedOutputCount::One),
            &super::super::EVAL_CATALOG_ENTRY,
        );
        assert!(eval.diagnostics.is_empty());
        assert_eq!(eval.outputs[0].kind, ValueKindFact::Unknown);
        assert!(eval.effects.0.contains(&EffectKind::WorkspaceWrite));

        let evalc = infer_evalc(
            &request(vec![source], RequestedOutputCount::Exactly(2)),
            &super::super::EVALC_CATALOG_ENTRY,
        );
        assert_eq!(evalc.outputs[0].kind, ValueKindFact::String);
        assert_eq!(evalc.outputs[1].kind, ValueKindFact::Unknown);
    }

    #[test]
    fn assignin_has_no_output_and_rejects_known_invalid_text() {
        let numeric = ValueFact::scalar(ValueKindFact::Logical);
        let result = infer_assignin(
            &request(
                vec![numeric.clone(), numeric.clone(), numeric],
                RequestedOutputCount::One,
            ),
            &super::super::ASSIGNIN_CATALOG_ENTRY,
        );
        assert!(result.diagnostics.len() >= 3);
        assert!(result.outputs.len() == 1); // The requested slot remains a diagnosed unknown fact.
    }
}
