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
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-EVALFR-ARITY",
            "evalfr expects one model and one point",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if let Some(model) = request.arguments.first() {
        let valid = match &model.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => {
                object
                    .runtime_class
                    .as_ref()
                    .is_some_and(|class| class == &standard::TRANSFER_FUNCTION.owned())
                    || (object.runtime_class.is_none() && object.class.is_none())
            }
            _ => false,
        };
        if !valid
            || model.shape.element_count().is_some_and(|n| n != 1)
            || matches!(model.residency, ResidencyFact::Remote { .. })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-EVALFR-MODEL",
                "sys must be a scalar SISO tf model",
                0,
            ));
        }
    }
    if let Some(point) = request.arguments.get(1) {
        if point.shape.element_count().is_some_and(|n| n != 1)
            || matches!(point.residency, ResidencyFact::Remote { .. })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-EVALFR-POINT",
                "x must be a scalar host numeric point",
                1,
            ));
        }
    }
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.shape = ShapeFact::Scalar;
    output.storage = StorageFact::Scalar;
    output.residency = ResidencyFact::Host;
    finish_fixed(entry, request, output, diagnostics)
}
