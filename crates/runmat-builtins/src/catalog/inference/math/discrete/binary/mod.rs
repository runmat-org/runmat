use crate::catalog::inference::argument_error;
use crate::{BinaryNumberTheoryRule, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, ValueFact,
};

mod class;
mod shape;

pub(super) fn infer(
    operation: BinaryNumberTheoryRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let output = match request.arguments.as_slice() {
        [left, right] => infer_output(left, right, &mut diagnostics),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-INTEGER-BINARY-ARITY",
                "gcd and lcm require exactly two inputs",
                request.arguments.len().min(2),
            ));
            ValueFact::unknown(DynamicReason::RuntimeValue)
        }
    };
    let count = match operation {
        BinaryNumberTheoryRule::Lcm => 1,
        BinaryNumberTheoryRule::Gcd => request.outputs.requested.known_count().unwrap_or(3).min(3),
    };
    if operation == BinaryNumberTheoryRule::Gcd
        && count >= 2
        && output
            .numeric()
            .and_then(|numeric| numeric.class.integer_class())
            .is_some_and(|integer| !integer.is_signed())
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-GCD-COEFFICIENT-CLASS",
            "gcd Bezout coefficients require double, single, or signed integer inputs",
            0,
        ));
    }
    finish(entry, request, vec![output; count], diagnostics)
}

fn infer_output(
    left: &ValueFact,
    right: &ValueFact,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> ValueFact {
    let class = class::resolve(left, right, diagnostics);
    let shape = shape::resolve(left, right, diagnostics);
    match (class, shape) {
        (Some(class), Some(shape)) => shape::numeric_output(class, shape),
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    }
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

#[cfg(test)]
mod tests;
