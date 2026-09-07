mod callback;
mod validation;

use crate::catalog::inference::finish_fixed;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, ResidencyFact, StorageFact, ValueFact,
};

pub(in crate::catalog) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = validation::arguments(request);
    let callback = callback::result(request);
    diagnostics.extend(callback.diagnostics);

    let mut output = callback
        .output
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::UnresolvedCallable));
    output.shape = validation::output_shape(request, &mut diagnostics);
    output.residency = ResidencyFact::Host;
    output.storage = if matches!(output.kind, runmat_types::ValueKindFact::Unknown) {
        StorageFact::Unknown
    } else {
        StorageFact::Dense
    };

    let mut inference = finish_fixed(entry, request, output, diagnostics);
    inference.effects.0.extend(callback.effects.0);
    inference.capabilities.0.extend(callback.capabilities.0);
    inference
}
