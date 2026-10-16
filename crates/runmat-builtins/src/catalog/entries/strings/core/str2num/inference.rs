use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, ValueFact, ValueKindFact,
};

fn infer_outputs(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    outputs: Vec<ValueFact>,
    diagnostics: Vec<runmat_types::InferenceDiagnostic>,
) -> CallInference {
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    inferred.diagnostics.extend(diagnostics);
    inferred
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !matches!(request.arguments.len(), 1 | 3) {
        diagnostics.push(argument_error(
            "RM-CATALOG-STR2NUM-ARITY",
            "str2num requires text and optionally Evaluation=method",
            0,
        ));
    }
    for index in 1..request.arguments.len().min(3) {
        if !matches!(
            request.arguments[index].kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-STR2NUM-EVALUATION",
                "Evaluation option and method must be scalar text",
                index,
            ));
        }
    }
    if let Some(input) = request.arguments.first() {
        if !matches!(
            input.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-STR2NUM-TEXT",
                "str2num requires scalar text",
                0,
            ));
        }
    }
    infer_outputs(
        request,
        entry,
        vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        diagnostics,
    )
}
