mod output;

use super::super::super::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{infer_call, CallContract, CallInference, CallRequest, ValueFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let requested = request.outputs.requested.known_count();
    if requested == Some(2) {
        return infer_two_outputs(request, entry);
    }

    let (value_output, mut diagnostics) =
        super::unary::infer_value(crate::LogarithmKind::Binary, request);
    let exponent_output = output::infer(
        request.arguments.first(),
        &mut diagnostics,
        requested.is_none(),
    );
    finish(
        entry,
        request,
        vec![value_output, exponent_output],
        diagnostics,
    )
}

fn infer_two_outputs(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let output = output::infer(request.arguments.first(), &mut diagnostics, true);
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-LOG2-ARITY",
            "log2 accepts exactly one input",
            1,
        ));
    }
    finish(entry, request, vec![output.clone(), output], diagnostics)
}

fn finish(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
    outputs: Vec<ValueFact>,
    mut diagnostics: Vec<runmat_types::InferenceDiagnostic>,
) -> CallInference {
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}
