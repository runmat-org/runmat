use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    standard, CallInference, CallRequest, DynamicReason, ResidencyFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-DCGAIN-ARITY",
            "dcgain expects exactly one transfer-function model",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if let Some(input) = request.arguments.first() {
        let valid_model = match &input.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => match object.runtime_class.as_ref() {
                Some(class) => class == &standard::TRANSFER_FUNCTION.owned(),
                None => object.class.is_none(),
            },
            _ => false,
        };
        if !valid_model
            || input.shape.element_count().is_some_and(|count| count != 1)
            || matches!(input.residency, ResidencyFact::Remote { .. })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-DCGAIN-MODEL",
                "sys must be a scalar SISO transfer-function object",
                0,
            ));
        }
    }
    // A real-coefficient model can still produce a complex indeterminate gain
    // after exact pole-zero cancellation. The scalar host shape is known, but
    // result class and domain must remain dynamic until runtime evaluation.
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.shape = ShapeFact::Scalar;
    output.storage = StorageFact::Scalar;
    output.residency = ResidencyFact::Host;
    finish_fixed(entry, request, output, diagnostics)
}
