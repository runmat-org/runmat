mod binary;
mod input;
mod unary;

#[cfg(test)]
mod tests;

use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, DynamicReason, ValueFact};

use super::super::super::{argument_error, finish_fixed};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let output = match request.arguments.as_slice() {
        [exponent] => unary::infer(exponent, &mut diagnostics),
        [significand, exponent] => binary::infer(significand, exponent, &mut diagnostics),
        arguments => {
            diagnostics.push(argument_error(
                "RM-CATALOG-POW2-ARITY",
                "pow2 requires one or two inputs",
                arguments.len().min(1),
            ));
            ValueFact::unknown(DynamicReason::RuntimeValue)
        }
    };
    finish_fixed(entry, request, output, diagnostics)
}
