use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    for (index, argument) in request.arguments.iter().enumerate() {
        if matches!(
            argument.residency,
            ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FINDOBJ-RESIDENCY",
                "search roots, controls, and property values must be host graphics metadata",
                index,
            ));
        }
    }
    let first = request.arguments.first();
    if let Some(first) = first {
        if matches!(
            first.kind,
            ValueKindFact::Logical
                | ValueKindFact::Callable(_)
                | ValueKindFact::Object(_)
                | ValueKindFact::Foreign(_)
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FINDOBJ-ROOT",
                "first argument must be a graphics root or a text predicate",
                0,
            ));
        }
        if first
            .numeric()
            .is_some_and(|number| number.domain == NumericDomain::Complex)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-FINDOBJ-ROOT",
                "graphics root handles must be real",
                0,
            ));
        }
    }
    let mut output = if diagnostics.is_empty() {
        let mut value = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::Unknown,
            StorageFact::Dense,
        );
        value.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
        value
    } else {
        ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
    };
    output.residency = ResidencyFact::Host;
    finish_fixed_outputs(entry, request, vec![output], diagnostics)
}
