mod callback;
mod invocation;
mod options;

use crate::{catalog::inference::finish_fixed, BuiltinCatalogEntry};
use runmat_types::{
    CallInference, CallRequest, CellFact, DynamicReason, ResidencyFact, StorageFact, ValueFact,
    ValueKindFact,
};

pub(in crate::catalog) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let plan = invocation::Plan::from_request(request);
    let mut diagnostics = plan.diagnostics;
    let callback = callback::infer(request, &plan.array_indices);
    diagnostics.extend(callback.diagnostics);

    let element = callback
        .output
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::UnresolvedCallable));
    let output = if plan.uniform_output == Some(false) {
        ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(element),
                elements: Vec::new(),
                elements_complete: false,
            }),
            plan.output_shape,
            StorageFact::Dense,
        )
    } else if plan.uniform_output == Some(true) {
        uniform_output(element, plan.output_shape, plan.has_device_input)
    } else {
        let mut unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
        unknown.shape = plan.output_shape;
        unknown
    };

    let mut inference = finish_fixed(entry, request, output, diagnostics);
    inference.effects.0.extend(callback.effects.0);
    inference.capabilities.0.extend(callback.capabilities.0);
    inference
}

fn uniform_output(
    mut element: ValueFact,
    shape: runmat_types::ShapeFact,
    has_device_input: bool,
) -> ValueFact {
    element.shape = shape;
    element.storage = if element.is_scalar() {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    element.residency = if has_device_input {
        ResidencyFact::Unknown
    } else {
        ResidencyFact::Host
    };
    element
}
