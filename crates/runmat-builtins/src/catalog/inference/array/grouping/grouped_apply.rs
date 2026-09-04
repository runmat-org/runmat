use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, ShapeFact, ValueFact,
    ValueKindFact,
};

use super::super::super::argument_error;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = validate(request);
    let requested = request.outputs.requested.known_count();
    let outputs = requested.map(|count| callback_outputs(request, count));
    let mut contract = outputs.map_or_else(
        || CallContract::dynamic(DynamicReason::RuntimeValue),
        CallContract::fixed,
    );
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}

fn validate(request: &CallRequest) -> Vec<runmat_types::InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-SPLITAPPLY-ARITY",
            "splitapply requires a function, at least one data input, and group numbers",
            request.arguments.len(),
        ));
    }
    if let Some(function) = request.arguments.first() {
        if !matches!(
            function.kind,
            ValueKindFact::Callable(_) | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-SPLITAPPLY-FUNCTION",
                "splitapply requires a callable first argument",
                0,
            ));
        }
    }
    diagnostics
}

fn callback_outputs(request: &CallRequest, count: usize) -> Vec<ValueFact> {
    let callable = request.arguments.first().and_then(|value| {
        if let ValueKindFact::Callable(callable) = &value.kind {
            Some(callable)
        } else {
            None
        }
    });
    (0..count)
        .map(|index| {
            let Some(output) = callable.and_then(|callable| callable.outputs.get(index)) else {
                return ValueFact::unknown(DynamicReason::UnresolvedCallable);
            };
            let mut output = output.clone();
            output.shape = grouped_shape(&output.shape);
            output
        })
        .collect()
}

fn grouped_shape(shape: &ShapeFact) -> ShapeFact {
    let Some(mut dimensions) = shape.known_dims() else {
        return shape
            .rank()
            .map_or(ShapeFact::Unknown, |rank| ShapeFact::Ranked { rank });
    };
    dimensions.resize(2, Some(1));
    dimensions[0] = None;
    ShapeFact::from(dimensions)
}
