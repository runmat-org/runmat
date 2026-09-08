mod callback;
mod invocation;
mod options;
mod output;

use crate::{catalog::inference::finish_fixed, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest, DynamicReason, ValueFact};

pub(in crate::catalog) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let plan = invocation::Plan::from_request(request);
    let mut diagnostics = plan.diagnostics;
    let callback = callback::infer(request, &plan.cell_indices, &plan.extra_indices);
    diagnostics.extend(callback.diagnostics);
    let element = callback
        .output
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::UnresolvedCallable));
    let output = match plan.uniform_output {
        Some(false) => output::nonuniform(element, plan.output_shape),
        Some(true) => output::uniform(element, plan.output_shape),
        None => {
            let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
            output.shape = plan.output_shape;
            output
        }
    };
    let mut inference = finish_fixed(entry, request, output, diagnostics);
    inference.effects.0.extend(callback.effects.0);
    inference.capabilities.0.extend(callback.capabilities.0);
    inference
}
