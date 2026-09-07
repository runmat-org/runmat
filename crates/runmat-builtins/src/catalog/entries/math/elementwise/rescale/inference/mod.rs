mod admission;
mod output;
mod validation;

use crate::catalog::inference::{finish_fixed, materialize};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, DynamicReason, ValueFact};

pub(in crate::catalog) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let admitted = admission::inspect(request);
    let mut validation_diagnostics = Vec::new();
    validation::arguments(&admitted, &mut validation_diagnostics);
    let mut diagnostics = admitted.diagnostics;
    diagnostics.extend(validation_diagnostics);
    let mut result = admitted.source.map_or_else(dynamic, |source| {
        output::fact(
            source,
            &admitted.operands,
            admitted.grammar_known,
            &mut diagnostics,
        )
    });
    materialize(&mut result);
    finish_fixed(entry, request, result, diagnostics)
}

fn dynamic() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}
